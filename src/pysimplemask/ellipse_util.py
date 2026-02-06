import numpy as np
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from matplotlib.patches import Ellipse
from skimage.measure import moments, moments_central


def find_ellipse_parameters(image):
    # Calculate moments for all non-zero pixels combined
    m = moments(image, order=1)
    if m[0, 0] == 0: return None
    
    # Centroid
    y0 = m[0, 1] / m[0, 0]
    x0 = m[1, 0] / m[0, 0]
    
    # Central moments for rotation and axes
    mu = moments_central(image, center=(y0, x0), order=2)
    
    # Covariance matrix components
    mu20 = mu[0, 2] / mu[0, 0] # Note: skimage moments use (row, col)
    mu02 = mu[2, 0] / mu[0, 0]
    mu11 = mu[1, 1] / mu[0, 0]
    
    # Calculate angle (orientation)
    # This matches the logic of regionprops but for the whole image
    orientation = 0.5 * np.arctan2(2 * mu11, mu02 - mu20)
    
    # Eigenvalues for axis lengths
    term = np.sqrt((mu02 - mu20)**2 + 4 * mu11**2)
    l1 = (mu02 + mu20 + term) / 2
    l2 = (mu02 + mu20 - term) / 2
    
    major_axis = 4 * np.sqrt(l1)
    minor_axis = 4 * np.sqrt(l2)

    return {
        "x0": x0, "y0": y0, 
        "angle": np.pi / 2 - orientation,
        "major_axis": major_axis, 
        "minor_axis": minor_axis
    }


def find_ellipse_parameters_connected(image):
    label_img = label(image)
    props = regionprops(label_img)
    target = props[0]
    y0, x0 = target.centroid

    orientation = target.orientation # radians, counter-clockwise
    major_axis = target.major_axis_length
    minor_axis = target.minor_axis_length

    angle = np.pi / 2 - orientation

    param = {
        "x0": x0,
        "y0": y0,
        "angle": angle,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "aspect_ratio": major_axis / minor_axis,
    }
    return param


def compute_ellipse_gradient(vgrid, hgrid, param):
    # 1. Transform grid to local coordinates
    ygrid_rel = vgrid - param["y0"] 
    hgrid_rel = hgrid - param["x0"] 

    # 2. Calculate local coordinates along the axes
    theta = param["angle"]  # Ensure this is in radians
    hgrid_local = hgrid_rel * np.cos(theta) + ygrid_rel * np.sin(theta)
    vgrid_local = -hgrid_rel * np.sin(theta) + ygrid_rel * np.cos(theta)

    # 3. Calculate the ellipse equation (radial component)
    a = param["major_axis"] / 2
    b = param["minor_axis"] / 2
    
    # normalized distance from center (1.0 = on the ellipse boundary)
    rho = np.sqrt((hgrid_local / a) ** 2 + (vgrid_local / b) ** 2)
    
    # 4. Calculate the polar angle (angular component)
    # arctan2(y, x) returns angle in radians from -pi to pi
    phi = np.arctan2(vgrid_local, hgrid_local)
    
    return rho, phi


def test01():
    # 1. Create a synthetic binary image
    image = np.zeros((500, 400), dtype=np.uint8)
    # ellipse(r, c, r_radius, c_radius, shape, rotation)
    rr, cc = ellipse(150, 200, 100, 50, rotation=np.deg2rad(30))
    image[rr, cc] = 1
    param = find_ellipse_parameters(image)

    vgrid, hgrid = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
    grad, phi = compute_ellipse_gradient(vgrid, hgrid, param)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(grad, cmap="jet")
    ax[2].imshow(phi, cmap="jet")
    plt.show()


if __name__ == "__main__":
    test01()
    
    