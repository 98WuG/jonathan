import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
import nrrd
import time

# Some constants
INPUT_FOLDER = 'raw_data'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


def load_scan(path):

    ct_slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path) if re.match(r'.*\.dcm', s)]
    pet_slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path) if re.match(r'IM_[0-9]+$', s)]
    ct_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    pet_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    for s in os.listdir(path):
        if re.match(r'.*GTV\.nrrd', s):
            segmentation = nrrd.read(path + '/' + s)

    try:
        try:
            ct_slice_thickness = np.abs(ct_slices[0].ImagePositionPatient[2] - ct_slices[1].ImagePositionPatient[2])
            pet_slice_thickness = np.abs(pet_slices[0].ImagePositionPatient[2] - pet_slices[1].ImagePositionPatient[2])
        except:
            ct_slice_thickness = np.abs(pet_slices[0].SliceLocation - pet_slices[1].SliceLocation)
            pet_slice_thickness = np.abs(pet_slices[0].SliceLocation - pet_slices[1].SliceLocation)
    except:
        ct_slice_thickness = ct_slices[0][0x18, 0x88].value
        pet_slice_thickness = pet_slices[0][0x18, 0x88].value

    for s in ct_slices:
        s.SliceThickness = ct_slice_thickness

    for s in pet_slices:
        s.SliceThickness = pet_slice_thickness

    return ct_slices, pet_slices, segmentation


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    # image_index = 600
    binary_image = np.array(image > -320, dtype=np.int8)

    # close small holes, like nose
    binary_image = scipy.ndimage.morphology.binary_closing(binary_image, structure=np.ones([1, 10, 10])) + 1

    # plt.figure()
    # plt.imshow(binary_image[image_index], cmap=plt.cm.gray)
    # plt.figure()
    # plt.imshow(closed_binary_image[image_index], cmap=plt.cm.gray)
    # plt.show()

    labels = measure.label(binary_image, connectivity=1)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # plt.figure()
    # plt.imshow(image[image_index], cmap=plt.cm.gray)
    # plt.figure()
    # plt.imshow(binary_image[image_index], cmap=plt.cm.gray)
    # plt.figure()
    # plt.imshow(labels[image_index], cmap=plt.cm.gray)
    # plt.show()

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


PIXEL_MEAN = 0.25


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def resample_mask(image, space_directions, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([space_directions[0, 0], space_directions[1, 1], space_directions[2, 2]], dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def load_image(path):
    # Start timer
    start_time = time.time()

    # Loading the data
    print('Loading data...')
    ct_data, pet_data, tumor_mask = load_scan(path)
    ct_pixel = get_pixels_hu(ct_data)
    pet_pixel = get_pixels_hu(pet_data)

    # Resampling the image
    print('Resampling mask...')
    tumor_resampled_mask, spacing = resample_mask(tumor_mask[0], tumor_mask[1]['space directions'])
    print('Resampling CT...')
    ct_resampled_image, spacing = resample(ct_pixel, ct_data, [1, 1, 1])
    print('Resampling PET...')
    pet_resampled_image, spacing = resample(pet_pixel, pet_data, [1, 1, 1])

    # Rotate the mask
    tumor_resampled_mask = np.transpose(tumor_resampled_mask, [2, 1, 0])

    # Get the lung mask
    print('Generating lung mask...')
    lung_mask = get_lung_mask(ct_resampled_image)

    # Print time
    print("--- Patient data loaded in %s seconds ---" % (time.time() - start_time))

    return ct_resampled_image, pet_resampled_image, tumor_resampled_mask, lung_mask


def get_lung_mask(image):

    # Get the segmentation
    segmented_lungs_fill = segment_lung_mask(image, True)

    # Dilate the mask
    dilated_mask = scipy.ndimage.morphology.binary_dilation(segmented_lungs_fill, structure=np.ones([10, 10, 10]))

    return dilated_mask


def display_ct_pet(dir):

    # Load the scans
    ct = np.load(dir + '/CT.npy')
    pet = np.load(dir + '/PET.npy')
    seg = np.load(dir + '/mask.npy')
    lung = np.load(dir + '/lung.npy')

    # Make the pet the same size as the ct scan
    difference = int((pet.shape[1] - ct.shape[1]) / 2)
    pet = pet[:, difference:-difference, difference:-difference]

    # Find where the tumor is
    z_max = np.unravel_index(np.argmax(seg), seg.shape)[0]
    print(z_max)
    image_index = z_max + 10
    print(ct.shape, pet.shape, seg.shape)

    # Plot all of them
    plt.subplot(2, 2, 3)
    plt.imshow(np.transpose([4*pet[image_index]/np.max(pet), seg[image_index], normalize(ct[image_index])/2.5], [1, 2, 0]))
    plt.subplot(2, 2, 1)
    plt.imshow(ct[image_index], cmap=plt.cm.gray)
    plt.subplot(2, 2, 2)
    plt.imshow(pet[image_index], cmap=plt.cm.gray)
    plt.subplot(2, 2, 4)
    plt.imshow(lung[image_index], cmap=plt.cm.gray)
    plt.show()


def process_data():
    # Start timer
    start_time = time.time()

    # Loop through every patient
    for patient in patients:
        print('Patient: ', patient)

        # Load the image of the given patient
        ct, pet, mask, lung = load_image(INPUT_FOLDER + "/" + patient)

        # Create a directory if it doesn't exist
        if not os.path.exists('processed_data/' + patient):
            os.makedirs('processed_data/' + patient)

        # Save as numpy arrays
        np.save('processed_data/' + patient + "/PET", pet)
        np.save('processed_data/' + patient + "/CT", ct)
        np.save('processed_data/' + patient + "/mask", mask)
        np.save('processed_data/' + patient + "/lung", lung)

    # Print time
    print("--- Total time elapsed: %s seconds ---" % (time.time() - start_time))

def main():

    process_data()
    # display_ct_pet('processed_data/Lung-CHOP-001')

    # print('Creating mask...')
    # lung_mask = get_lung_mask(loaded_image)
    # print('Completed creating mask')
    #
    # np.save('DL_test/test_segment.npy', lung_mask)
    #
    # print('Plotting mask...')
    # plt.imshow(lung_mask[0], cmap=plt.cm.gray)
    # plt.show()

    # display_ct_pet('processed_data/20100506/CT.npy', 'processed_data/20100506/PET.npy', 600)
    #






    # first_patient = load_scan(INPUT_FOLDER + "/" + patients[0])
    # first_patient_pixels = get_pixels_hu(first_patient)
    # segmented_lungs_fill = np.load('DL_test/segmented_lungs_fill.npy')
    # segmented_lungs = np.load('DL_test/segmented_lungs.npy')

    # pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    # print("Shape before resampling\t", first_patient_pixels.shape)
    # print("Shape after resampling\t", pix_resampled.shape)
    # np.save('DL_test', pix_resampled)
    # print('saved')

    # segmented_lungs = segment_lung_mask(pix_resampled[:800], False)
    # segmented_lungs_fill = segment_lung_mask(pix_resampled[:800], True)
    # np.save('DL_test/segmented_lungs', segmented_lungs)
    # np.save('DL_test/segmented_lungs_fill', segmented_lungs_fill)

    # plot_3d(segmented_lungs_fill, 0)




if __name__ == "__main__":
    main()