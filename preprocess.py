import numpy as np # linear algebra
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
# from skimage import measure, morphology
import re
import nrrd
import time
# import raster_geometry as mrt


def load_scan(path):
    """
    Loads the ct scan, pet scans, and tumor mask
    :param path: Folder which represents a patient
    :return: ct, pet, tumor databases
    """

    # .dcm files are the ct scan
    ct_slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path) if re.match(r'.*\.dcm', s)]

    # Files with IM_XXX and no extension are the PET scan
    pet_slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path) if re.match(r'IM_[0-9]+$', s)]

    # Sort them so it's in order
    ct_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    pet_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Find the tumor mask
    for s in os.listdir(path):
        if re.match(r'.*GTV\.nrrd', s):
            segmentation = nrrd.read(path + '/' + s)

    # A bunch of ways to find the z-axis spacing RIP
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

    # Make sure they remember their z-axis spacing
    for s in ct_slices:
        s.SliceThickness = ct_slice_thickness

    for s in pet_slices:
        s.SliceThickness = pet_slice_thickness

    return ct_slices, pet_slices, segmentation


def get_pixels_hu(slices):
    """
    Gets the pixels from the database things
    :param slices: list of databases representing a DICOM scan
    :return: NumPy array of pixel values
    """
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # --- NOT SURE IF I NEED THIS ---
    # # Set outside-of-scan pixels to 0
    # # The intercept is usually -1024, so air is approximately 0
    # image[image == -2000] = 0

    # Convert to the numbers they should be, DICOM is stupid
    for slice_number in range(len(slices)):

        # Get the intercept and slope
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        # Change the slope
        image[slice_number] = slope * image[slice_number].astype(np.float64)
        image[slice_number] = image[slice_number].astype(np.int16)

        # Add the intercept
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    """
    This resamples the image as the new spacing in mm
    :param image: NumPy array of pixel intensities
    :param scan: List of databases (used to find original spacing)
    :param new_spacing: List representing new spacing
    :return: resampled_image, new_spacing
    """
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    # How much we need to scale each dimension
    resize_factor = spacing / new_spacing

    # Find the new shape
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)

    # Round the resize factor
    real_resize_factor = new_shape / image.shape

    # New spacing
    new_spacing = spacing / real_resize_factor

    # Actually use interpolation to create new image (fuck it takes long)
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def resample_mask(image, new_spacing=[1, 1, 1]):
    """
    This resamples the mask into a new dimension, needed a new one because these are .nrrd
    :param image: np array image
    :param space_directions: the spacing basically
    :param new_spacing: duh
    :return: resampled image
    """
    # Determine current pixel spacing
    spacing = np.array([1, 1, 1], dtype=np.float32)

    # How much we need to scale each dimension
    resize_factor = spacing / new_spacing

    # Find the new shape
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)

    # Round the resize factor
    real_resize_factor = new_shape / image.shape

    # New spacing
    new_spacing = spacing / real_resize_factor

    # Actually use interpolation to create new image (it takes long)
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def largest_label_volume(im, bg=-1):
    """
    Finds the largest volume in a scan of the same value
    :param im: 3D NumPy array representing image
    :param bg: int that will be ignored as background
    :return: largest volume label
    """

    # Finds the unique labels and the amount of times they pop up
    vals, counts = np.unique(im, return_counts=True)

    # Gets rid of background counts
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    # If there is a large volume, return it, otherwise it's all background
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

# I don't think I need this
# def segment_lung_mask(image, fill_lung_structures=True):
#     """
#     This returns a mask which covers the lungs
#     :param image: 3d np array
#     :param fill_lung_structures: true if you want to include hard structures inside lungs
#     :return: binary mask
#     """
#
#     # Makes all the air pixels 0, everything else 1
#     binary_image = np.array(image > -320, dtype=np.int8) + 1
#
#     # # Close up sinuses so the lungs are not connected to the outside
#     # binary_image = scipy.ndimage.morphology.binary_closing(binary_image, structure=mrt.sphere(25, 10)) + 1
#
#     # Turns every group of same digits into a unique number, separating air in lungs from air outside lungs
#     labels = measure.label(binary_image, connectivity=1)
#
#     # Pick the pixel in the very corner to determine which label is air.
#     #   Improvement: Pick multiple background labels from around the patient
#     #   More resistant to "trays" on which the patient lays cutting the air
#     #   around the person in half
#     background_label = labels[0, 0, 0]
#
#     # Fill the air around the person
#     binary_image[background_label == labels] = 2
#
#     # Method of filling the lung structures (that is superior to something like
#     # morphological closing)
#     if fill_lung_structures:
#
#         # For every slice we determine the largest solid structure
#         for i, axial_slice in enumerate(binary_image):
#
#             # Make axial slices binary, 0 is air enclosed in body, 1 is body and outside air
#             axial_slice = axial_slice - 1
#
#             # Label the air inside the body
#             labeling = measure.label(axial_slice)
#
#             # Find the largest volume (which will be everything except for the mass inside the lungs)
#             l_max = largest_label_volume(labeling, bg=0)
#
#             # If the slice contains lungs, the labels that aren't the maximum are turned to 1, which is solid in lungs
#             if l_max is not None:
#                 binary_image[i][labeling != l_max] = 1
#
#     # Make the lungs 1, everything else 0
#     binary_image -= 1
#     binary_image = 1 - binary_image
#
#     # # Remove other air pockets inside body
#     # labels = measure.label(binary_image, background=0)
#     # l_max = largest_label_volume(labels, bg=0)
#     # if l_max is not None:
#     #     binary_image[labels != l_max] = 0
#
#     # Dilate the image
#     dilated = scipy.ndimage.morphology.binary_dilation(binary_image, structure=mrt.sphere(15, 5))
#
#     return dilated


def normalize(image, min_bound=-1000.0, max_bound=400.0):
    """
    This normalizes the pixels between 0.0 and 1.0, where there is a min and max bound which cuts it off. This is
    useful for getting rid of bones in CT scans essentially.
    :param image: 3d np array
    :param min_bound: minimum value
    :param max_bound: maximum value
    :return: normalized image
    """
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def zero_center(image, pixel_mean=0.25):
    """
    This is to center the data mean around 0. This just helps
    :param image: image array
    :param pixel_mean: the mean
    :return: centered array
    """
    image = image - pixel_mean
    return image


def load_image(path):
    """
    This takes a patient folder and loads the CT, PET, tumor mask, and lung segmentation, all resampled to [1,1,1]
    :param path: duh
    :return: above
    """

    # Loading the data
    print('Loading data...')
    ct_data, pet_data, tumor_mask = load_scan(path)
    ct_pixel = get_pixels_hu(ct_data)
    pet_pixel = get_pixels_hu(pet_data)

    # Resampling the image
    print('Resampling mask...')
    tumor_resampled_mask1, spacing = resample_mask(tumor_mask[0], tumor_mask[1]['space directions'])
    tumor_resampled_mask2, spacing = resample_mask(tumor_mask[0], tumor_mask[1]['space directions'], [2, 2, 2])
    tumor_resampled_mask3, spacing = resample_mask(tumor_mask[0], tumor_mask[1]['space directions'], [4, 4, 4])
    print('Resampling CT...')
    ct_resampled_image, spacing = resample(ct_pixel, ct_data, [1, 1, 1])
    print('Resampling PET...')
    pet_resampled_image, spacing = resample(pet_pixel, pet_data, [1, 1, 1])

    # Rotate the mask
    tumor_resampled_mask1 = np.transpose(tumor_resampled_mask1, [2, 1, 0])
    tumor_resampled_mask2 = np.transpose(tumor_resampled_mask2, [2, 1, 0])
    tumor_resampled_mask3 = np.transpose(tumor_resampled_mask3, [2, 1, 0])

    # # Get the lung mask
    # print('Generating lung mask...')
    # lung_mask = segment_lung_mask(ct_resampled_image[:-200], True)

    return ct_resampled_image, pet_resampled_image, tumor_resampled_mask1, tumor_resampled_mask2, tumor_resampled_mask3


def process_data(parent_directory):
    """
    This takes a parent directory and processes all the patients, as folders, within it
    :param parent_directory:
    :return: none
    """
    # Start timer
    start_time = time.time()

    # Get the list of folders in patients
    patients = sorted(os.listdir(parent_directory))

    # Loop through every patient
    for patient in patients:
        print('Patient: ', patient)

        # Start timer
        start_time1 = time.time()

        # Load the image of the given patient
        ct, pet, mask1, mask2, mask3 = load_image(parent_directory + "/" + patient)

        # Create a directory if it doesn't exist
        if not os.path.exists('processed_data/' + patient):
            os.makedirs('processed_data/' + patient)

        # Save as numpy arrays
        np.save('processed_data/' + patient + "/PET", pet)
        np.save('processed_data/' + patient + "/CT", ct)
        np.save('processed_data/' + patient + "/mask_original", mask1)
        np.save('processed_data/' + patient + "/mask_half", mask2)
        np.save('processed_data/' + patient + "/mask_quarter", mask3)

        # np.save('processed_data/' + patient + "/lung", lung)

        # Print time
        print("--- Patient data loaded in %s seconds ---" % (time.time() - start_time1))

    # Print time
    print("--- Total time elapsed: %s seconds ---" % (time.time() - start_time))


def get_mask_bounds(mask):
    """
    Takes in a tumor mask, returns the upper and lower bounds
    :param mask: 3d np array
    :return: lower(x,y,z), upper(x,y,z)
    """

    # Looking for the x bounds
    x_mask = np.max(mask, 2)
    x_mask = np.max(x_mask, 1)
    x_index = [i for i, x in enumerate(x_mask) if x]
    x_up = x_index[-1]
    x_low = x_index[0]

    # Looking for the y bounds
    y_mask = np.max(mask, 2)
    y_mask = np.max(y_mask, 0)
    y_index = [i for i, x in enumerate(y_mask) if x]
    y_up = y_index[-1]
    y_low = y_index[0]

    # Looking for the z bounds
    z_mask = np.max(mask, 1)
    z_mask = np.max(z_mask, 0)
    z_index = [i for i, x in enumerate(z_mask) if x]
    z_up = z_index[-1]
    z_low = z_index[0]

    return [x_low, y_low, z_low], [x_up, y_up, z_up]


def cut_random_cubes(ct, pet, mask):
    """
    This cuts random 128x128x128 cubes that include the tumor
    :param ct: full ct scan
    :param pet: full pet scan
    :param mask: full mask
    :return: images (128, 128, 128, 2), mask (128, 128, 128), mask(64,64,64), mask(32,32,32)
    """

    # Make the pet scan the same size as the other scans
    difference = int((pet.shape[1] - ct.shape[1]) / 2)
    pet = pet[:, difference:-difference, difference:-difference]

    # Get the high and low indices of the mask and turn into numpy arrays
    low, high = get_mask_bounds(mask)
    low = np.array(low)
    high = np.array(high)
    difference = high - low

    # Check if the tumor can be put inside with padding of at least 12
    if np.any(difference > 104):
        print('Tumor too big')

    # This is the index with the tumor right in the corner with 12 padding
    temp_index = np.array(low) - 12

    # How much more can you subtract from it without getting out of bounds
    max_subtraction = 104 - difference

    # Add on a number from 0 to max_addition
    subtraction = np.floor((max_subtraction + 1) * np.random.uniform(size=[3, ]))
    final_index = (temp_index - subtraction).astype(int)

    # Use the index to find the resulting cubes
    final_ct = ct[final_index[0]:final_index[0]+128, final_index[1]:final_index[1]+128, final_index[2]:final_index[2]+128]
    final_pet = pet[final_index[0]:final_index[0]+128, final_index[1]:final_index[1]+128, final_index[2]:final_index[2]+128]
    final_mask = mask[final_index[0]:final_index[0]+128, final_index[1]:final_index[1]+128, final_index[2]:final_index[2]+128]

    # Downsample the mask 2, 4
    mask2, spacing = resample_mask(final_mask, [2, 2, 2])
    mask3, spacing = resample_mask(final_mask, [4, 4, 4])

    return final_ct, final_pet, final_mask, mask2, mask3


def display_ct_pet_processed(ct, pet, seg, seg1):
    """
    This displaces the ct, pet, tumor mask, and lung mask in 2d axial view of the tumor
    :param folder: the folder in which all these scans, saved as np arrays, are located
    :return: none
    """

    print("CT shape: ", ct.shape, " || Pet shape: ", pet.shape, " || Seg shape: ", seg.shape, " || Seg1 shape: ", seg1.shape)

    # Find where the tumor is
    image_index = np.unravel_index(np.argmax(seg), seg.shape)[0] + 5
    print('Z index: ', image_index)

    # Plot all of them
    plt.subplot(2, 2, 3)
    plt.imshow(
        np.transpose([4 * pet[image_index] / np.max(pet), seg[image_index], normalize(ct[image_index]) / 2.5],
                     [1, 2, 0]))
    plt.subplot(2, 2, 1)
    plt.imshow(ct[image_index], cmap=plt.cm.gray)
    plt.subplot(2, 2, 2)
    plt.imshow(pet[image_index], cmap=plt.cm.gray)
    plt.subplot(2, 2, 4)
    plt.imshow(seg1[image_index], cmap=plt.cm.gray)
    plt.show()


def display_ct_pet(folder):
    """
    This displaces the ct, pet, tumor mask, and lung mask in 2d axial view of the tumor
    :param folder: the folder in which all these scans, saved as np arrays, are located
    :return: none
    """

    # Load the scans
    ct = np.load(folder + '/CT.npy')
    pet = np.load(folder + '/PET.npy')
    seg = np.load(folder + '/mask_original.npy')
    seg1 = np.load(folder + '/mask_quarter.npy')
    print("CT shape: ", ct.shape, " || Pet shape: ", pet.shape, " || Seg shape: ", seg.shape, " || Seg1 shape: ",
          seg1.shape)

    # Make the pet the same size as the ct scan
    difference = int((pet.shape[1] - ct.shape[1]) / 2)
    pet = pet[:, difference:-difference, difference:-difference]

    # Find where the tumor is
    image_index = np.unravel_index(np.argmax(seg), seg.shape)[0] + 10
    print('Z index: ', image_index)

    # Plot all of them
    plt.subplot(2, 2, 3)
    plt.imshow(
        np.transpose([4 * pet[image_index] / np.max(pet), seg[image_index], normalize(ct[image_index]) / 2.5],
                     [1, 2, 0]))
    plt.subplot(2, 2, 1)
    plt.imshow(ct[image_index], cmap=plt.cm.gray)
    plt.subplot(2, 2, 2)
    plt.imshow(pet[image_index], cmap=plt.cm.gray)
    plt.subplot(2, 2, 4)
    plt.imshow(seg1[int(image_index / 4)], cmap=plt.cm.gray)
    plt.show()


def main():

    # Processes all patients folders within the given directory
    # process_data('C:/Users/Jonathan Lee/Dropbox/Lung PET segmentation/VA PET/')

    # Display the data in the given folder
    display_ct_pet('processed_data/Lung-VA-001')

    ct = np.load('processed_data/Lung-VA-001/CT.npy')
    pet = np.load('processed_data/Lung-VA-001/PET.npy')
    mask = np.load('processed_data/Lung-VA-001/mask_original.npy')

    final_ct, final_pet, final_mask, mask2, mask3 = cut_random_cubes(ct, pet, mask)

    display_ct_pet_processed(final_ct, final_pet, final_mask, mask3)


if __name__ == "__main__":
    main()