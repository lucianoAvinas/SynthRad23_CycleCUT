import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from pathlib import Path
from shutil import rmtree
from scipy.ndimage import zoom
from sklearn.mixture import GaussianMixture


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_name', default='data_out', 
                        help='Name output data folder. Will overwrite if folder already exists')

    parser.add_argument('--apply_mask', action='store_true', 
    	                help='Apply provided mask. All values outside the mask will be '
    	                     'set to the minimum value found in the volume. May introduce '
    	                     'discontinuity artifacts at mask boundary.')

    parser.add_argument('--directions', default='z', choices=['x', 'y', 'z', 'all'],
                        help='Determines which slice directions to save to numpy. Selection '
                             '"all" will save x, y, and z directions for a 2.5D approach.')

    parser.add_argument('--sites', default='both', choices=['both', 'brain', 'pelvis'],
                        help='Anatomy sites to use during training. Currently only brain, pelvis, '
                             'or both sites are available.')

    parser.add_argument('--bias_correct', default='none', choices=['none', 'n4', 'lap'],
                        help='Bias correction algorithm to use on volume. If "lap" is selected '
                             'with normalization "gm", then joint method LapGM will be used.')

    parser.add_argument('--normalization', default='none', choices=['none', 'max', 'zscore', 'gm'],
                        help='Normalization algorithm to use on volume. If "gm" is selected '
                             'with bias correction "lap", then joint method LapGM will be used.')

    return parser.parse_args()


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

    for site in sites:
        data_loc = task_root / site
        site_let = site[0].upper()
        for site_id, file_path in enumerate(data_loc.iterdir()):
            if file_path.name == 'overview':
                continue

            print(f'At {site} site processing volume: {file_path.name}')

            mr_img = nib.load(file_path / 'mr.nii.gz').get_fdata()
            ct_img = nib.load(file_path / 'ct.nii.gz').get_fdata()

            if args.apply_mask:
                mask = (nib.load(file_path / 'mask.nii.gz').get_fdata()) == 0

                mr_img[mask] = mr_img.min()
                ct_img[mask] = ct_img.min()

            if args.bias_correct == 'n4':
                mr_img = n4_debias(mr_img)

            elif args.bias_correct == 'lap':
                #algo = ...
                raise NotImplementedError()

            if args.normalization == 'max':
                mr_img = mr_img / mr_img.max()

            elif args.normalization == 'zscore':
                mr_img = mr_img - mr_img.mean()
                mr_img = mr_img / mr_img.std()

            elif args.normalization == 'gm':
                if args.bias_correct == 'lap':
                    # re-use algo result here
                    raise NotImplementedError()
                else:
                    gm_model = GaussianMixture(n_components=5) ## n_components subject to change
                    gm_model.fit(mr_img.flatten()[:,None])
                    mr_img = mr_img / gm_model.means_.max()

            for direc, order in axs:
                for i, (mr_slice, ct_slice) in enumerate(zip(mr_img.transpose(order), 
                                                         ct_img.transpose(order))):
                    np.save(save_mr / f'{site_id}{site_let}_{i}_{direc}_A.npy', mr_slice)
                    np.save(save_ct / f'{site_id}{site_let}_{i}_{direc}_B.npy', ct_slice)          
