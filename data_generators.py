import numpy as np
from glob import glob
from pydicom import dcmread
from cv2 import imread

from augmentations import augmentations
from config import paths


def get_paths(dataset, modality):

    if modality in ('CT', 'ALL'):
        ct_dcm_paths = paths[dataset] + '/**/CT/**/**.dcm'
        ct_grd_paths = paths[dataset] + '/**/CT/**/*.png'
        ct_dicom_list = sorted(glob(ct_dcm_paths, recursive=True))
        ct_ground_list = sorted(glob(ct_grd_paths, recursive=True))
    if modality in ('MR', 'ALL'):
        mr_dcm_in = paths[dataset] + '/**/MR/**/InPhase/*.dcm'
        mr_dcm_out = paths[dataset] + '/**/MR/**/OutPhase/*.dcm'
        mr_grd_t1 = paths[dataset] + '/**/MR/**/T1DUAL/**/*.png'
        mr_dcm_t2 = paths[dataset] + '/**/MR/**/T2SPIR/**/*.dcm'
        mr_grd_t2 = paths[dataset] + '/**/MR/**/T2SPIR/**/*.png'
        mr_dcm_in_list = sorted(glob(mr_dcm_in, recursive=True))
        mr_dcm_out_list = sorted(glob(mr_dcm_out, recursive=True))
        mr_dcm_t2_list = sorted(glob(mr_dcm_t2, recursive=True))
        mr_grd_t1_list = sorted(glob(mr_grd_t1, recursive=True))
        mr_grd_t2_list = sorted(glob(mr_grd_t2, recursive=True))
        mr_dicom_list = mr_dcm_in_list + mr_dcm_out_list + mr_dcm_t2_list
        mr_ground_list = mr_grd_t1_list + mr_grd_t1_list + mr_grd_t2_list

    if modality == 'CT':
        if dataset in ('train', 'eval'):
            assert len(ct_dicom_list) == len(ct_ground_list)  # Check lists length
            data_path_list = list(zip(ct_dicom_list, ct_ground_list))
        else:
            data_path_list = list(ct_dicom_list)
    elif modality == 'MR':
        if dataset in ('train', 'eval'):
            assert len(mr_dicom_list) == len(mr_ground_list)  # Check lists length
            data_path_list = list(zip(mr_dicom_list, mr_ground_list))
        else:
            data_path_list = list(mr_dicom_list)
    else:
        if dataset in ('train', 'eval'):
            assert len(ct_dicom_list + mr_dicom_list) == len(ct_ground_list + mr_ground_list)  # Check lists length
            data_path_list = list(zip(ct_dicom_list + mr_dicom_list, ct_ground_list + mr_ground_list))
        else:
            data_path_list = list(ct_dicom_list + mr_dicom_list)

    return data_path_list


# noinspection PyUnboundLocalVariable
def test_gen(params):
    data_paths = get_paths(dataset='infer', modality=params['modality'])
    for dcm_path in data_paths:
        dicom = dcmread(dcm_path).pixel_array
        dicom = (dicom - np.mean(dicom)) / np.std(dicom)
        yield dicom, dcm_path


def data_gen(dataset, args, only_paths=False):
    data_paths = get_paths(dataset=dataset, modality=args.modality)
    if only_paths:
        for dicom_path, label_path in data_paths:
            yield dicom_path, label_path
    else:
        if args.mode == 'train':
            np.random.shuffle(data_paths)
        if args.mode in ('train', 'eval'):
            for dicom_path, label_path in data_paths:
                dicom, label = dcmread(dicom_path).pixel_array, imread(label_path, 0)
                if args.modality == 'MR':
                    label[label != 63] = 0
                if dataset == 'train' and args.augm:  # Data augmentation
                    if np.random.random() < 0.5:
                        dicom, label = augmentations(dcm_image=dicom, grd_image=label)
                if args.modality in ('MR', 'ALL'):
                    if args.modality == 'MR':
                        resize = 320 - dicom.shape[0]
                    else:
                        resize = 512 - dicom.shape[0]
                    dicom = np.pad(dicom, [int(resize / 2)], mode='constant', constant_values=np.min(dicom))
                    label = np.pad(label, [int(resize / 2)], mode='constant', constant_values=0)
                dicom = (dicom - np.mean(dicom)) / np.std(dicom)  # Normalize
                label[label > 0] = 1
                yield dicom, label
        else:
            for dicom_path in data_paths:
                dicom = dcmread(dicom_path).pixel_array
                yield (dicom - np.mean(dicom)) / np.std(dicom), dicom_path  # Normalize


if __name__ == '__main__':
    a = get_paths('eval', 'ALL')
    for i in a:
        print(i)