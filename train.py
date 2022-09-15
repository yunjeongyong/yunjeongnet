import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


from option.config import Config
from model.yunswin import SwinTransformerV2
from model.backbone import resnet50_backbone
from inference import train_epoch, eval_epoch



# config file
# config file
config = Config({
    # dataset path
    'gpu_id': "0",
    "db_name": "KADID-10k",
    "train_csv_path": "C:\\Users\\yunjeongyong\\Desktop\\intern\\IQA\\iqa-db\\all_data_csv\\KADID-10k.csv",
    "train_data_path": "C:\\Users\\yunjeongyong\\Desktop\\intern\\IQA\\iqa-db\\",
    "train_txt_file_name": "./data/pipal21_train.txt",
    "val_txt_file_name": "./data/pipal21_val.txt",

    # optimization
    "scale_1": 384,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "n_epoch": 10,
    "val_freq": 1,
    "T_max": 50,
    "eta_min": 0,
    "num_avg_val": 5,
    "num_workers": 0,
    'momentum': 0.9,

    # model
    "patches_resolution":(1, 1),
    "patch_size": 8,
    "img_size": 224,
    "embed_dim": 768,
    "dim_mlp": 768,
    "num_heads": [4, 4],
    "window_size": 4,
    "depths": [2, 2],
    "num_outputs": 1,
    "num_tab": 2,
    "scale": 0.13,

    # load & save checkpoint
    "model_name": "model_maniqa",
    "output_path": "./output",
    "snap_path": "./output/models/",  # directory for saving checkpoint
    "log_path": "./output/log/maniqa/",
    "log_file": ".txt",
    "tensorboard_path": "./output/tensorboard/",
    'checkpoint': None,
})

# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')


# data selection
if config.db_name == 'KADID-10k':
    from data.kadid_10k import KadidDataset

# # dataset separation (8:2)
# train_scene_list, test_scene_list = RandShuffle(config)
# print('number of train scenes: %d' % len(train_scene_list))
# print('number of test scenes: %d' % len(test_scene_list))

train_dataset = KadidDataset(
    # db_path=config.db_path,
    # txt_file_name=config.txt_file_name,
    # scale_1=config.scale_1,
    # scale_2=config.scale_2,
    # transform=transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), RandHorizontalFlip(), ToTensor()]),
    # train_mode=True,
    # scene_list=train_scene_list,
    # train_size=config.train_size
    config.train_csv_path,
    config.train_data_path,
    scale_1=config.scale_1,
    img_size=None,
    is_train=True,
)

# data load
testset = KadidDataset(
    # db_path=config.db_path,
    # txt_file_name=config.txt_file_name,
    # scale_1=config.scale_1,
    # scale_2=config.scale_2,
    # transform=transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), RandHorizontalFlip(), ToTensor()]),
    # train_mode=True,
    # scene_list=train_scene_list,
    # train_size=config.train_size
    config.train_csv_path,
    config.train_data_path,
    scale_1=config.scale_1,
    img_size=None,
    is_train=False,
)


# testset = KadidDataset(config.train_csv_path,
#                        config.train_data_path,
#                        img_size=None,
#                        scale_1=2048,
#                        is_train=False)
# test_dataset = KadidDataset(
#     db_path=config.db_path,
#     txt_file_name=config.txt_file_name,
#     scale_1=config.scale_1,
#     scale_2=config.scale_2,
#     transform= transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensor()]),
#     train_mode=False,
#     scene_list=test_scene_list,
#     train_size=config.train_size
# )

train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)


# create model
model_backbone = resnet50_backbone().to(config.device)
model_transformer = SwinTransformerV2().to(config.device)


# loss function & optimization
criterion = torch.nn.L1Loss()
params = list(model_backbone.parameters()) + list(model_transformer.parameters())
optimizer = torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)


# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)


# train & validation
for epoch in range(start_epoch, config.n_epoch):
    loss, rho_s, rho_p = train_epoch(config, epoch, model_transformer, model_backbone, criterion, optimizer, scheduler, train_loader)

    if (epoch+1) % config.val_freq == 0:
        loss, rho_s, rho_p = eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader)

