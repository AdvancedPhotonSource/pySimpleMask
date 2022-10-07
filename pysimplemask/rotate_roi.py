import numpy as np
import skimage.io as skio
from skimage.morphology import binary_erosion
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
plt.style.use('science')


def create_circular_roi(N=1024, phi_range_deg=30):
    x = np.arange(N) - N // 2
    grid = np.zeros((N, N), dtype=bool)
    gx, gy = np.meshgrid(x, x)  
    radius = np.hypot(gx, gy)
    angle = np.arctan2(gx, gy)
    angle_roi = np.logical_and(angle > 0, angle < np.deg2rad(phi_range_deg))
    radius_roi = np.logical_and(radius > 360, radius < 480)
    roi = radius_roi * angle_roi
    skio.imsave('roi_%d_deg.tif' % phi_range_deg, roi)
    plt.imshow(roi)
    plt.show()
    


def to_2d_img(shape, vh, crop=False):
    roi = np.zeros(shape, dtype=np.uint8)
    for n in range(len(vh[0])):
        roi[vh[0, n], vh[1, n]] += 1

    if crop:
        vmin, vmax = np.min(vh[0]), np.max(vh[0]) + 1
        hmin, hmax = np.min(vh[1]), np.max(vh[1]) + 1
        roi = roi[vmin:vmax, hmin:hmax]
    return roi


def find_nearby_vaccant(shape, roi, pos, neighbour=8):
    # find a nearby point that has zero ancestor
    v, h = pos[0], pos[1]
    
    # four neighbours
    nlist = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if neighbour == 8:
        nlist += [(-1, -1),  (-1, 1), (1, -1), (1, 1)]

    total = 0
    new_pos = None
    for t in nlist:
        va = t[0] + v
        ha = t[1] + h
        # in the image
        if shape[0] > va >= 0 and shape[1] > ha >= 0:
            if roi[va, ha] == 0:
                total += 1
                new_pos = (va, ha)
                break

    if total == 0:
        return None
    else:
        return new_pos


def fix_double_pixels(shape, vh, neighbour=8):
    roi = np.zeros(shape, dtype=np.uint8)
    record = {}

    # convert the list of pixels to a 2d image and record the positions that
    # have 2 ancestors 
    for n in range(len(vh[0])):
        roi[vh[0, n], vh[1, n]] += 1
        if roi[vh[0, n], vh[1, n]] == 2:
            record[tuple(vh[:, n])] = n

    idx_two = np.where(roi == 2)
    for n in range(len(idx_two[0])):
        orig_pos = (idx_two[0][n], idx_two[1][n])
        pos = find_nearby_vaccant(shape, roi, orig_pos, neighbour)
        if pos is not None:
            idx = record[orig_pos]
            vh[:, idx] = np.array([pos[0], pos[1]])
            roi[pos[0], pos[1]] = 1
            roi[orig_pos[0], orig_pos[1]] = 1
    
    return vh 


def rotate_without_alias(roi, center, angle, mask=None):
    # v0, h0 is the center of the x-ray beam
    if center is None:
        center = (roi.shape[0] // 2, roi.shape[1] // 2)

    # original center of mass of the object
    com = center_of_mass(roi)

    # unwrap angle
    angle_deg = np.rad2deg(angle)
    # make it positive, in the range of [0, 360) deg
    angle_deg = angle_deg - np.floor(angle_deg / 360) * 360

    # angle_deg is in [-45, 45) now;
    while angle_deg > 45:
        angle_deg -= 90 
        roi = np.rot90(roi)

    angle_rel = np.deg2rad(angle_deg)

    vh = np.array(np.nonzero(roi)).astype(np.float64)   # 2 x n

    mean_val = np.mean(vh, axis=1)
    vh = (vh.T - mean_val).T

    rot_mat0 = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])
    
    rot_mat1 = np.array([
        [1, np.tan(-angle_rel / 2.0)],
        [0, 1]])

    rot_mat2 = np.array([
        [1, 0],
        [np.sin(angle_rel), 1]])

    # vertical shear
    vh = rotate_and_correct(vh, rot_mat1, direction='v')
    # horizontal shear
    vh = rotate_and_correct(vh, rot_mat2, direction='h')
    # # vertical shear
    vh = rotate_and_correct(vh, rot_mat1, direction='v')

    # compute the new center of mass
    new_center = np.array(com) - np.array(center)
    new_center = np.matmul(rot_mat0, new_center) + np.array(center)
    new_center = np.floor(new_center + 0.5).astype(np.int64)

    vh = (vh.T + new_center).T
    vh = vh.astype(np.int64)
    # roi_rot = to_2d_img(None, vh[0], vh[1], True)
    assert mask.shape == shape
    

    return vh


def rotate_and_correct(vh, rot_mat, direction='v'):
    vh2 = np.matmul(rot_mat, vh)    # 2 x n
    vh2_int = np.floor(vh2 + 0.5)

    # recover the unsheared direction to preserve the resolution
    if direction == 'v':
        vh2_int[1] = vh2[1]
    elif direction == 'h':
        vh2_int[0] = vh2[0]

    return vh2_int


def main2(angle=2*np.pi/3.0, center=None, roi=None):
    if roi is None:
        roi = skio.imread('roi_30_deg.tif')
        roi = roi.astype(bool)

    shape = roi.shape

    vh = rotate_without_alias(roi, center, angle)

    roi_raw = to_2d_img(shape, vh, crop=False)
    new_com = center_of_mass(roi_raw)
    print('new_center_of_mass', new_com)
    print('total_1', np.sum(roi > 0))
    print('total_2', np.sum(roi_raw > 0))

    roi_raw = roi + roi_raw
    two_count = np.sum(roi_raw == 2)
    plt.imshow(roi_raw, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('rot_angle: %3d deg, twos = %d' % (np.rad2deg(angle), two_count))
    plt.savefig('rot_angle_%03d_deg' % np.rad2deg(angle), dpi=600)
    # plt.show()
    plt.close()
    return 0


def main(angle=2*np.pi/3.0, center=None, roi=None):
    if roi is None:
        roi = skio.imread('roi.tif')
        roi = roi.astype(bool)

    shape = roi.shape
    vh = np.array(np.nonzero(roi))     # 2 x n

    # v0, h0 is the center of the x-ray beam
    if center is None:
        center = (roi.shape[0] // 2, roi.shape[1] // 2)

    center = np.array(center).astype(np.float64)    # (2,)

    vh_f = (vh.astype(np.float64).T - center).T     # 2 x n

    rot_mat = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])

    vh_rot = np.matmul(rot_mat, vh_f)   # 2 x 2, 2 x n -> 2 x n
    vh_rot = (vh_rot.T + center).T      # 2 x n

    # cast the coordinates onto the integer grid 
    vh = np.floor(vh_rot + 0.5).astype(np.int64)

    # fix the double-ancestor pixels
    vh = fix_double_pixels(shape, vh, 4)
    # can be called multiple times
    vh = fix_double_pixels(shape, vh, 4)

    roi_raw = to_2d_img(shape, vh, crop=False)
    two_count = np.sum(roi_raw == 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(roi_raw, vmin=0, vmax=2)
    plt.colorbar()
    plt.title('rot_angle: %3d deg, twos = %d' % (np.rad2deg(angle), two_count))
    plt.savefig('rot_angle_%03d_deg' % np.rad2deg(angle), dpi=600)
    plt.show()
    plt.close()
    return two_count

    vf, hf = to_2d(choice)
    roi2 = np.zeros_like(roi)
    roi2[(vf, hf)] = 1
    # print(len(record), np.sum(roi2 > 0), np.sum(roi > 0))
    # plt.imshow(roi + roi2)
    # plt.imshow(roi2)
    # plt.show()
    roi_raw = to_2d_img(shape, vf, hf, crop=True)
    plt.imshow(roi_raw, vmin=0, vmax=2)
    plt.colorbar()
    plt.title('rot_angle: %3d deg' % np.rad2deg(angle))
    plt.savefig('rot_angle_2_%03d_deg' % np.rad2deg(angle), dpi=600)
    plt.close()
    # return

# def plot_data():
#     data = np.loadtxt('num_twos_as_function_of_phi.txt')
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(data[:, 0], data[:, 1], '.-')
#     ax.set_xlabel('$\\phi$ (degree)')
#     ax.set_ylabel('Numer of Pixels Twos')
#     plt.savefig('fig1.png', dpi=600)
#     plt.show()


# def plot_data_study2():
#     data = np.loadtxt('num_twos_as_function_of_center.txt')
#     fig, ax = plt.subplots(1, 1)

#     x_range = (np.min(data[:, 1]), np.max(data[:, 1]))
#     y_range = (np.min(data[:, 0]), np.max(data[:, 0]))
#     # ax.plot(data[:, 0], data[:, 1], '.-')
#     im = ax.imshow(data[:, 2].reshape(64, 64), extent=(*x_range, *y_range), cmap=plt.cm.jet)
#     ax.set_ylabel('Vertical Center (pixel)')
#     ax.set_xlabel('Horizontal Center (pixel)')
#     plt.colorbar(im, ax=ax)
#     plt.savefig('fig2.png', dpi=600)
#     plt.show()


# def study2():
#     data = []
#     angle = np.pi / 4.0
#     xlist = np.linspace(-1.0, 1.0, 64)
#     for v in xlist:
#         print(v)
#         for h in xlist:
#             cen = (516 / 2 + v, 1556 / 2 + h)
#             y = main(angle, center=cen)
#             data.append([*cen, y])

#     data = np.array(data) 
#     np.savetxt('num_twos_as_function_of_center.txt', data)


# def study3():
#     roi = skio.imread('roi.tif')
#     roi = roi.astype(bool)
#     angle = np.pi / 4.0

#     data = []
#     while True:
#         y = main(angle, roi=roi)
#         total = np.sum(roi)
#         ratio = y / total 
#         print(total, ratio)
#         data.append([total, ratio])

#         roi = binary_erosion(roi)
#         if np.sum(roi) <= 0:
#             break

#     data = np.array(data) 
#     np.savetxt('num_twos_as_function_of_roi_size.txt', data)


# def plot_data_study3():
#     data = np.loadtxt('num_twos_as_function_of_roi_size.txt')
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(data[:, 0], data[:, 1], '.-', ms=2.0)
#     ax.set_xlabel('Original ROI size (pixels)')
#     ax.set_ylabel('Numer of Pixels Twos')
#     plt.savefig('fig3.png', dpi=600)
#     plt.show()




if __name__ == '__main__':
    # data = []
    # for angle in np.linspace(0, np.pi * 2, 361):
    #     y = main(angle)
    #     data.append([angle, y])
    
    # data = np.array(data)
    # data[:, 0] = np.rad2deg(data[:, 0])
    # np.savetxt('num_twos_as_function_of_phi.txt', data)
    for n in range(0, 365, 5):
        y = main2(np.deg2rad(n))
    # y = main2(np.deg2rad(60))

    # main()
    # create_circular_roi(N=1024, phi_range_deg=30)