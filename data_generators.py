import numpy as np
from glob import glob
from pydicom import dcmread
from cv2 import imread

from augmentations import augmentations
from config import paths


def get_paths(dataset, modality):
    ct_dcm_paths = paths[dataset] + '/**/CT/**/**.dcm'
    ct_grd_paths = paths[dataset] + '/**/CT/**/*.png'
    mr_dcm_in = paths[dataset] + '/**/MR/**/InPhase/*.dcm'
    mr_dcm_out = paths[dataset] + '/**/MR/**/OutPhase/*.dcm'
    mr_grd_t1 = paths[dataset] + '/**/MR/**/T1DUAL/**/*.png'
    mr_dcm_t2 = paths[dataset] + '/**/MR/**/T2SPIR/**/*.dcm'
    mr_grd_t2 = paths[dataset] + '/**/MR/**/T2SPIR/**/*.png'

    ct_dicom_list = sorted(glob(ct_dcm_paths, recursive=True))
    ct_ground_list = sorted(glob(ct_grd_paths, recursive=True))

    mr_dcm_in_list = sorted(glob(mr_dcm_in, recursive=True))
    mr_dcm_out_list = sorted(glob(mr_dcm_out, recursive=True))
    mr_dcm_t2_list = sorted(glob(mr_dcm_t2, recursive=True))
    mr_grd_t1_list = sorted(glob(mr_grd_t1, recursive=True))
    mr_grd_t2_list = sorted(glob(mr_grd_t2, recursive=True))

    mr_dicom_list = mr_dcm_in_list + mr_dcm_out_list + mr_dcm_t2_list
    mr_ground_list = mr_grd_t1_list + mr_grd_t1_list + mr_grd_t2_list

    if modality == 'CT':
        dicom_list = ct_dicom_list
        ground_list = ct_ground_list
    elif modality == 'MR':
        dicom_list = mr_dicom_list
        ground_list = mr_ground_list
    else:
        dicom_list = ct_dicom_list + mr_dicom_list
        ground_list = ct_ground_list + mr_ground_list

    if dataset in ('train', 'eval'):
        assert len(dicom_list) == len(ground_list)  # Check lists length
        zipped_list = list(zip(dicom_list, ground_list))
        return zipped_list
    if dataset == 'infer':
        return dicom_list


# noinspection PyUnboundLocalVariable
def test_gen(params):
    data_paths = get_paths(dataset='infer', modality=params['modality'])
    for dcm_path in data_paths:
        dicom = dcmread(dcm_path).pixel_array
        dicom = (dicom - np.mean(dicom)) / np.std(dicom)
        yield dicom, dcm_path


def data_gen(dataset, args, only_paths=False):
    data_paths = get_paths(dataset=dataset, modality=args.modality)
    if args.shuffle:
        np.random.shuffle(data_paths)
    if only_paths:
        for dicom_path, label_path in data_paths:
            yield dicom_path, label_path
    else:
        for dicom_path, label_path in data_paths:
            dicom, label = dcmread(dicom_path).pixel_array, imread(label_path, 0)
            if args.modality == 'MR':
                label[label != 63] = 0
            if dataset == 'train':  # Data augmentation
                if np.random.random() < args.augm_prob:
                    dicom, label = augmentations(dcm_image=dicom, grd_image=label, augm_set=args.augm_set)

                if args.modality in ('MR', 'ALL'):
                    if args.modality == 'MR':
                        resize = 320 - dicom.shape[0]
                    else:
                        resize = 512 - dicom.shape[0]
                    dicom = np.pad(dicom, [int(resize / 2)], mode='constant', constant_values=np.min(dicom))
                    label = np.pad(label, [int(resize / 2)], mode='constant', constant_values=np.min(label))

            dicom = (dicom - np.mean(dicom)) / np.std(dicom)  # Normalize
            label[label > 0] = 1
            yield dicom, label