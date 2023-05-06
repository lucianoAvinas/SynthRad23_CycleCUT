import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import itertools
from torch.utils.checkpoint import checkpoint

# try:
#     from apex import amp
# except ImportError:
#     print("Please install NVIDIA Apex for safe mixed precision if you want to use non default --opt_level")

class CycleCUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            

        parser.set_defaults(pool_size=0)  # no image pooling



        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A_GAN', 'D_A_real', 'D_A_fake', 'G', 'NCE_A', 'cycle_A', 
        'G_B_GAN', 'D_B_real', 'D_B_fake', 'NCE_B', 'cycle_B']
        # self.loss_names = ['G_A_GAN', 'D_A_real', 'D_A_fake', 'G_A', 'NCE_A']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_A','rec_A','rec_B','idt_A','idt_B']
        # self.visual_names_A = ['real_A', 'fake_B', 'rec_A']
        # self.visual_names_B = ['real_B', 'fake_A', 'rec_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y_A', 'NCE_Y_B']
            # self.loss_names += ['NCE_Y_B']
            # self.visual_names_A += ['idt_B']
            # self.visual_names_B += ['idt_A']

        if self.isTrain:
            self.model_names = ['G_A', 'F_A', 'D_A', 'G_B', 'F_B', 'D_B']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']

        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt, not opt.no_tanh)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt, not opt.no_tanh)
        self.netF_A = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_B = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            # self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # if opt.apex:
            #     [self.netG_A, self.netG_B, self.netD_A, self.netD_B], [self.optimizer_G, self.optimizer_D] = amp.initialize(
            #         [self.netG_A, self.netG_B, self.netD_A, self.netD_B], [self.optimizer_G, self.optimizer_D], opt_level=opt.opt_level, num_losses=3)

        # need to be wrapped after amp.initialize
        # self.make_data_parallel()

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_A_loss().backward()                  # calculate gradients for D
            self.compute_D_B_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF_A.parameters(), self.netF_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                # self.optimizer_F = torch.optim.Adam(self.netF_A.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        # self.set_requires_grad(self.netD_A, True)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.loss_D_A = self.compute_D_A_loss()
        self.loss_D_B = self.compute_D_B_loss()
        self.loss_D_A.backward()
        self.loss_D_B.backward()
        self.optimizer_D.step()

        # update G
        # self.set_requires_grad(self.netD_A, False)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake_B = self.netG_A(self.real_A)
        # self.fake_B = self.fake_B[:self.real_A.size(0)]
        # self.fake_B = self.netG_A(self.real_A)  # G_A(A)

        # if not self.isTrain or not self.opt.checkpointing:
        #     self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # else:
        #     self.fake_B = checkpoint(self.netG_A, self.real_A)

        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        
        if not self.isTrain or not self.opt.checkpointing:
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        else:
            self.rec_A = checkpoint(self.netG_B, self.fake_B)
        
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.fake_A = self.netG_B(self.real_B)
        # self.fake_A = self.fake_A[:self.real_B.size(0)]

        # if not self.isTrain or not self.opt.checkpointing:
        #     self.fake_A = self.netG_B(self.real_B)  # G_A(A)
        # else:
        #     self.fake_A = checkpoint(self.netG_B, self.real_B)

        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        if not self.isTrain or not self.opt.checkpointing:
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        else:
            self.rec_B = checkpoint(self.netG_A, self.fake_A)

        if self.opt.nce_idt:
            # self.idt_B = self.fake[self.real_A.size(0):]
            # self.idt_A = self.netG_A(self.real_A)
            # self.idt_B = self.netG_B(self.real_B)
            if not self.isTrain or not self.opt.checkpointing:
                self.idt_A = self.netG_A(self.real_A)
            else:
                self.idt_A = checkpoint(self.netG_A, self.real_A)

            # self.idt_B = self.netG_B(self.real_A)
            if not self.isTrain or not self.opt.checkpointing:
                self.idt_B = self.netG_B(self.real_B)
            else:
                self.idt_B = checkpoint(self.netG_B, self.real_B)

    def compute_D_A_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD_A(fake)
        self.loss_D_A_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD_A(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_A_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D_A= (self.loss_D_A_fake + self.loss_D_A_real) * 0.5
        return self.loss_D_A

    def compute_D_B_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_A.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD_B(fake)
        self.loss_D_B_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD_B(self.real_A)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_B_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D_B = (self.loss_D_B_fake + self.loss_D_B_real) * 0.5
        return self.loss_D_B

    # def compute_D_loss(self):
    #     """Calculate GAN loss for the discriminator"""
    #     fake = self.fake_B.detach()
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     pred_fake = self.netD(fake)
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
    #     # Real
    #     self.pred_real = self.netD(self.real_B)
    #     loss_D_real = self.criterionGAN(self.pred_real, True)
    #     self.loss_D_real = loss_D_real.mean()

    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake_B = self.fake_B
        fake_A = self.fake_A
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake_B = self.netD_A(fake_B)
            pred_fake_A = self.netD_B(fake_A)
            self.loss_G_A_GAN = self.criterionGAN(pred_fake_B, True).mean() * self.opt.lambda_GAN
            self.loss_G_B_GAN = self.criterionGAN(pred_fake_A, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A_GAN = 0.0
            self.loss_G_B_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE_A = self.calculate_NCE_A_loss(self.real_A, self.fake_B)
            self.loss_NCE_B = self.calculate_NCE_B_loss(self.real_B, self.fake_A)
        else:
            self.loss_NCE_A, self.loss_NCE_A_bd = 0.0, 0.0
            self.loss_NCE_B, self.loss_NCE_B_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y_A = self.calculate_NCE_A_loss(self.real_B, self.idt_B)
            self.loss_NCE_Y_B = self.calculate_NCE_B_loss(self.real_A, self.idt_A)
            loss_NCE_both = (self.loss_NCE_A + self.loss_NCE_Y_A + self.loss_NCE_B + self.loss_NCE_Y_B) * 0.5
            # loss_NCE_both = (self.loss_NCE_B + self.loss_NCE_Y_B) * 0.5
        else:
            loss_NCE_both = self.loss_NCE_A + self.loss_NCE_B

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        # self.loss_G_A = self.loss_G_A_GAN
        # self.loss_G_B = self.loss_G_B_GAN

        # self.loss_G = self.loss_G_A_GAN + self.loss_G_B_GAN + loss_NCE_both
        self.loss_G = self.loss_G_A_GAN + self.loss_G_B_GAN + self.loss_cycle_A + self.loss_cycle_B + loss_NCE_both
        # self.loss_G = self.loss_G_A_GAN + loss_NCE_both

        return self.loss_G

    def calculate_NCE_A_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF_A(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF_A(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_NCE_B_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF_B(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF_B(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # def calculate_NCE_loss(self, src, tgt):
    #     n_layers = len(self.nce_layers)
    #     feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

    #     if self.opt.flip_equivariance and self.flipped_for_equivariance:
    #         feat_q = [torch.flip(fq, [3]) for fq in feat_q]

    #     feat_k = self.netG(src, self.nce_layers, encode_only=True)
    #     feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
    #     feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

    #     total_nce_loss = 0.0
    #     for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
    #         loss = crit(f_q, f_k) * self.opt.lambda_NCE
    #         total_nce_loss += loss.mean()

    #     return total_nce_loss / n_layers
