import lapgm
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk

lapgm.use_gpu(True)

from pathlib import Path
from shutil import rmtree
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_name', default='data_out', 
                        help='Name output data folder. Will overwrite if folder already exists')

    parser.add_argument('--apply_mask', action='store_true', 
    	                help='Apply provided mask. All values outside the mask will be '
    	                     'set to the minimum value found in the volume. May introduce '
    	                     'discontinuity artifacts at mask boundary.')

    parser.add_argument('--directions', default='x', choices=['x', 'y', 'z', 'all'],
                        help='Determines which slice directions to save to numpy. Selection '
                             '"all" will save x, y, and z directions for a 2.5D approach.')

    parser.add_argument('--sites', default='both', choices=['both', 'brain', 'pelvis'],
                        help='Anatomy sites to use during training. Currently only brain, pelvis, '
                             'or both sites are available.')

    parser.add_argument('--bias_correct', default='none', choices=['none', 'n4', 'lap'],
                        help='Bias correction algorithm to use on volume. If "lap" is selected '
                             'with normalization "gm", then joint method LapGM will be used.')

    parser.add_argument('--normalization', default='none', choices=['none', 'max', 'zscore', 'gm', 'nyul'],
                        help='Normalization algorithm to use on volume. If "gm" is selected '
                             'with bias correction "lap", then joint method LapGM will be used.')

    parser.add_argument('--volume_out', action='store_true', 
                        help='Save out full numpy volume instead of slices.')

    parser.add_argument('--pad_mode', default='constant', choices=['constant', 'edge', 'reflect'],
                        help='Decides padding mode to use when padding images to same size')

    parser.add_argument('--div_4', action='store_true', 
                        help='Makes sure excess padding is divisible by 4.')

    return parser.parse_args()


"""Nyul & Udupa piecewise linear histogram matching normalization
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 02 Jun 2021
"""

class NyulNormalize:
## Pulled and modified from https://github.com/jcreinhold/intensity-normalization
    def __init__(self, output_min_value=1.0, output_max_value=100.0, min_percentile=1.0,
                 max_percentile=99.0, percentile_after_min=10.0, percentile_before_max=90.0,
                 percentile_step=10.0):
        """Nyul & Udupa piecewise linear histogram matching normalization

        Args:
            output_min_value: where min-percentile mapped for output normalized image
            output_max_value: where max-percentile mapped for output normalized image
            min_percentile: min percentile to account for while finding
                standard histogram
            max_percentile: max percentile to account for while finding
                standard histogram
            next_percentile_after_min: next percentile after min for finding
                standard histogram (percentile-step creates intermediate percentiles)
            prev_percentile_before_max: previous percentile before max for finding
                standard histogram (percentile-step creates intermediate percentiles)
            percentile_step: percentile steps between next-percentile-after-min and
                 prev-percentile-before-max for finding standard histogram
        """
        super().__init__()
        self.output_min_value = output_min_value
        self.output_max_value = output_max_value
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.percentile_after_min = percentile_after_min
        self.percentile_before_max = percentile_before_max
        self.percentile_step = percentile_step
        self._percentiles = None
        self.standard_scale = None

    def _get_voi(self, image, mask=None):
        return image.flatten() if mask is None else image[mask].flatten()

    def normalize_image(self, image, mask=None):
        voi = self._get_voi(image, mask)

        landmarks = self.get_landmarks(voi)
        if self.standard_scale is None:
            msg = "This class must be fit before being called."
            raise ValueError
        f = interp1d(landmarks, self.standard_scale, fill_value="extrapolate")
        normalized = f(voi)
        return normalized

    @property
    def percentiles(self):
        if self._percentiles is None:
            percs = np.arange(
                self.percentile_after_min,
                self.percentile_before_max + self.percentile_step,
                self.percentile_step,
            )
            _percs = ([self.min_percentile], percs, [self.max_percentile])
            self._percentiles = np.concatenate(_percs)
        assert isinstance(self._percentiles, np.ndarray)
        return self._percentiles

    def get_landmarks(self, image):
        landmarks = np.percentile(image, self.percentiles)
        return landmarks  # type: ignore[return-value]

    def _fit(self, images, masks=None):
        """Compute standard scale for piecewise linear histogram matching

        Args:
            images: set of NifTI MR image paths which are to be normalized
            masks: set of corresponding masks (if not provided, estimated)
        """
        n_percs = len(self.percentiles)
        standard_scale = np.zeros(n_percs)
        n_images = len(images)
        if masks is not None and n_images != len(masks):
            raise ValueError("There must be an equal number of images and masks.")
        elif masks is None:
            masks = [None] * n_images

        for i, (image, mask) in enumerate(zip(images, masks)):
            voi = self._get_voi(image, mask)
            landmarks = self.get_landmarks(voi)
            min_p = np.percentile(voi, self.min_percentile)
            max_p = np.percentile(voi, self.max_percentile)
            f = interp1d([min_p, max_p], [self.output_min_value, self.output_max_value])
            landmarks = np.array(f(landmarks))
            standard_scale += landmarks

        self.standard_scale = standard_scale / n_images


def n4_debias(data, mask=None, zm_fctr=0.5, bias_fwhm=0.15, conv_thresh=0.001, max_iters=100, 
              n_fitlevels=4, n_cntpnts=4, n_histbins=200, spline_order=3, w_filtnoise=0.01):
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetBiasFieldFullWidthAtHalfMaximum(bias_fwhm)
    corrector.SetConvergenceThreshold(conv_thresh)
    corrector.SetMaximumNumberOfIterations([max_iters]*n_fitlevels)
    corrector.SetNumberOfControlPoints([n_cntpnts]*3)
    corrector.SetNumberOfHistogramBins(n_histbins)
    corrector.SetSplineOrder(spline_order)
    corrector.SetWienerFilterNoise(w_filtnoise)

    use_mask = mask is not None

    # only take t1 biased data
    if zm_fctr < 1:
        data_zm = zoom(data, zm_fctr)
        if use_mask:
            mask_zm = zoom(mask, zm_fctr)
    else:
        data_zm = data
        mask_zm = mask

    data_zm = sitk.GetImageFromArray(data_zm.astype(np.float32))
    if use_mask:
        mask_zm = sitk.GetImageFromArray(mask_zm.astype(np.uint8))
        corrector.Execute(data_zm, mask_zm)
    else:
        corrector.Execute(data_zm)

    sitk_logbias = corrector.GetLogBiasFieldAsImage(data_zm)
    np_logbias = sitk.GetArrayFromImage(sitk_logbias).astype(np.float32)

    if zm_fctr < 1:
        np_logbias = zoom(np_logbias, 1/zm_fctr)

        dim_diff = tuple(o-n for o,n in zip(data.shape, np_logbias.shape))
        # pad smaller dims
        np_logbias = np.pad(np_logbias, tuple((0, max(dim, 0)) for dim in dim_diff), 'edge')
        # crop bigger dims
        np_logbias = np_logbias[tuple([slice(0, dim if dim < 0 else None) for dim in dim_diff])]

    mask = data != 0
    data[mask] = data[mask] / np.exp(np_logbias[mask])

    return data


def pad_img(img, shp, mode):
    pad_ln = []
    for d1,d2 in zip(img.shape, shp):
        diff = d2 - d1
        pad_ln.append((diff//2, diff//2 + diff % 2))

    return np.pad(img, pad_ln, mode)


if __name__ == '__main__':
    args = parse_args()

    task_root = Path('datasets') / 'Task1'
    sites = ['brain', 'pelvis'] if args.sites == 'both' else [args.sites]

    axes_opts = dict(x=(0,1,2), y=(1,0,2), z=(2,0,1))
    axs = axes_opts.items() if args.directions == 'all' else [(args.directions, 
                                                               axes_opts[args.directions])]

    save_root = Path('datasets') / args.save_name
    save_mr = save_root / 'trainA'
    save_ct = save_root / 'trainB'

    if save_root.exists():
        rmtree(save_root)

    save_root.mkdir()
    save_mr.mkdir()
    save_ct.mkdir()

    BRAIN_MAX_SHP = (280, 284, 264) if args.div_4 else (280, 284, 262)
    PELVIS_MAX_SHP = (588, 412, 156) if args.div_4 else (586, 410, 153)

    CT_MN = -722.48
    CT_SD = 513.261

    if args.sites == 'both':
        MAX_SHP = tuple(max(d1,d2) for d1,d2 in zip(BRAIN_MAX_SHP, PELVIS_MAX_SHP))
    else:
        MAX_SHP = BRAIN_MAX_SHP if args.sites == 'brain' else PELVIS_MAX_SHP

    if not args.volume_out:
        temp_shp = [0, 0]
        for _, (_, ix, iy) in axs:
            temp_shp[0] = max(temp_shp[0], MAX_SHP[ix])
            temp_shp[1] = max(temp_shp[1], MAX_SHP[iy])
        MAX_SHP = tuple(temp_shp)

    for site in sites:
        data_loc = task_root / site
        site_let = site[0].upper()

        if args.normalization == 'nyul':
            nyul = NyulNormalize()
            imgs, masks = [], []
            for file_path in data_loc.iterdir():
                if file_path.name == 'overview':
                    continue
                imgs.append(nib.load(file_path / 'mr.nii.gz').get_fdata())
                masks.append((nib.load(file_path / 'mask.nii.gz').get_fdata()) == 1 if \
                              args.apply_mask else None)
            nyul._fit(imgs, masks)

        if args.bias_correct == 'lap':
            debias_obj = lapgm.LapGM()
            debias_obj.set_hyperparameters(tau=5e-4, n_classes=4, log_initialize=False)

        for site_id, file_path in enumerate(data_loc.iterdir()):
            if file_path.name == 'overview':
                continue

            print(f'At {site} site processing volume: {file_path.name}')

            mr_img = nib.load(file_path / 'mr.nii.gz').get_fdata()
            ct_img = nib.load(file_path / 'ct.nii.gz').get_fdata()

            ct_img = (ct_img - CT_MN) / CT_SD

            if args.apply_mask:
                mask = (nib.load(file_path / 'mask.nii.gz').get_fdata()) == 0

                mr_img[mask] = mr_img.min()
                ct_img[mask] = ct_img.min()
            else:
                mask = None

            if args.bias_correct == 'n4':
                mr_img = n4_debias(mr_img, mask)

            elif args.bias_correct == 'lap':
                mr_data = lapgm.to_sequence_array([mr_img])
                params = debias_obj.estimate_parameters(mr_data, max_em_iters=50)
                mr_img = lapgm.debias(mr_data, params).squeeze().astype(np.float64)

            if args.normalization == 'max':
                mr_mask = mr_img if mask is None else mr_img[mask]
                mr_img = mr_img / mr_mask.max()

            elif args.normalization == 'zscore':
                mr_mask = mr_img if mask is None else mr_img[mask]
                mr_img = mr_img - mr_mask.mean()
                mr_img = mr_img / mr_mask.std()

            elif args.normalization == 'gm':
                if args.bias_correct == 'lap':
                    mr_img = mr_img / np.exp(params.mu.max())
                else:
                    gm_model = GaussianMixture(n_components=4)
                    gm_model.fit(mr_img.flatten()[:,None])
                    mr_img = mr_img / gm_model.means_.max()

            elif args.normalization == 'nyul':
                if args.apply_mask:
                    mask = ~mask
                    mr_img[mask] = nyul.normalize_image(mr_img, mask)
                else:
                    mr_img = nyul.normalize_image(mr_img).reshape(mr_img.shape)

            if args.volume_out:
                mr_img = pad_img(mr_img, MAX_SHP, args.pad_mode)[None]
                ct_img = pad_img(ct_img, MAX_SHP, args.pad_mode)[None]

                np.save(save_mr / f'{site_id}{site_let}_{direc}_A.npy', mr_img)
                np.save(save_ct / f'{site_id}{site_let}_{direc}_A.npy', ct_img)
            else:
                for direc, order in axs:
                    for i, (mr_slice, ct_slice) in enumerate(zip(mr_img.transpose(order), 
                                                             ct_img.transpose(order))):
                        mr_slice = pad_img(mr_slice, MAX_SHP, args.pad_mode)[None]
                        ct_slice = pad_img(ct_slice, MAX_SHP, args.pad_mode)[None]

                        np.save(save_mr / f'{site_id}{site_let}_{i}_{direc}_A.npy', mr_slice)
                        np.save(save_ct / f'{site_id}{site_let}_{i}_{direc}_B.npy', ct_slice)          
