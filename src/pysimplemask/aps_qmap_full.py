import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.sparse import coo_matrix, diags
import skimage.io as skio
import time


TRANSMISSION_QMAP_UNIT = {
    "q": "Å⁻¹",
    "phi": "deg",
    "qx": "Å⁻¹",
    "qy": "Å⁻¹",
    "qr": "Å⁻¹",
    "alpha": "deg",
    "x": "pixel",
    "y": "pixel",
    "none": "Unit",
}


REFLECTION_QMAP_UNIT = {
    "q": "Å⁻¹",
    "phi": "deg",
    "qx": "Å⁻¹",
    "qy": "Å⁻¹",
    "qz": "Å⁻¹",
    "qr": "Å⁻¹",
    "tth": "deg",
    "alpha": "deg",
    "chi": "deg",
    "x": "pixel",
    "y": "pixel",
    "none": "Unit",
}


def convert_to_serializable_object(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple, np.ndarray)):
        obj = [convert_to_serializable_object(y) for y in obj]
        return obj
    elif isinstance(obj, dict):
        new_x = {}
        for k, v in obj.items():
            new_x[k] = convert_to_serializable_object(v)
        return new_x
    else:
        return obj


def convert_to_hashable_str(obj):
    return json.dumps(convert_to_serializable_object(obj))


def to_single_precision(qmap):
    for k, v in qmap.items():
        if v.dtype == np.float64:
            qmap[k] = v.astype(np.float32)
        elif v.dtype in [np.int64, np.uint64]:
            qmap[k] = v.astype(np.uint32)
    return qmap


def to_levels(qmap, levels=36):
    minval, maxval = np.min(qmap), np.max(qmap)
    delta = (maxval - minval) / levels
    if delta == 0:
        return qmap
    qmap_level = (qmap - minval) / delta
    qmap_level = qmap_level.astype(np.uint8)
    return qmap_level


def rotate_vector(vector, axis=0, angle_deg=0.0):
    """
    Rotate a 3D vector around one of the principal axes.

    Parameters:
    - vector (np.ndarray): 3D vector of shape (3,)
    - axis (int): Axis to rotate around: 0 (x), 1 (y), or 2 (z)
    - angle_deg (float): Rotation angle in degrees

    Returns:
    - np.ndarray: Rotated 3D vector
    """
    vector = np.asarray(vector, dtype=float)

    if angle_deg == 0.0:
        return vector

    if axis not in (0, 1, 2):
        raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z)")

    angle = np.deg2rad(angle_deg)

    if axis == 0:
        rot_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
    elif axis == 1:
        rot_mat = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
    else:  # axis == 2
        rot_mat = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    return rot_mat @ vector


def qmap_container_from_meta(meta):
    det_position_m = [meta["det_y"], meta["det_z"], meta["det_distance"]]
    sample_rotate_deg = [-meta["alpha_incident_deg"], 0, 0]
    sg_obj = ScatteringGeometry(
        meta["det_shape"],
        meta["beam_center"],
        det_position_m,
        sample_rotate_deg=sample_rotate_deg,
        energy_kev=meta["energy"],
        pixel_size_m=meta["det_pixel_size"],
        sg_type="reflection",
    )
    return sg_obj


class ScatteringGeometryManager:
    def __init__(self) -> None:
        self.cache = {}

    def get_qmap(self, meta_dict):
        mid = convert_to_hashable_str(meta_dict)
        if mid not in self.cache:
            print("create new qmap")
            self.cache[mid] = qmap_container_from_meta(meta_dict)
        else:
            print("use cached qmap")
        return self.cache[mid]


class ScatteringGeometry:
    def __init__(
        self,
        det_shape,
        beam_center,  # in pixels
        det_position_m,
        sg_type="reflection",
        energy_kev=8.0,
        det_rotate_deg=(0, 0, 0),
        sample_rotate_deg=(0, 0, 0),
        pixel_size_m=75e-3,
    ) -> None:

        self.shape = np.array(det_shape, dtype=np.uint64)
        self.mask = np.ones(self.shape, dtype=bool)
        self.beam_center = beam_center
        self.pixel_size_m = pixel_size_m
        self.energy_kev = energy_kev

        self.sg_type = sg_type
        self.qmap_unit = self.populate_qmap_unit(sg_type)

        self.incident_angle_deg = 0.0
        self.qmap = {}
        self.remap_solver = {}
        self.lincut_solver_cache = {}
        self.current_correction_config = None
        self.correction_mat = np.ones(self.shape, dtype=np.float64)
        self.position_args = (det_shape, beam_center, pixel_size_m)

        self.vector, self.origin = self.get_base_vector(
            det_rotate_deg, sample_rotate_deg, det_position_m
        )

        self.position, qmap = self.calculate_positions(oversampling=1)

        self.qmap.update(qmap)
        self.qmap.update(self.calculate_solid_angle(self.position))
        self.qmap.update(self.calculate_momentum_transfer(self.position))

        # convert all angles from radian to degree
        for key, val in self.qmap_unit.items():
            if val == "deg":
                self.qmap[key] = np.rad2deg(self.qmap[key])

        self.qmap_level = self.qmap.copy()
        for k, v in self.qmap_level.items():
            self.qmap_level[k] = to_levels(v)

    def get_base_vector(self, det_rotate_deg, sample_rotate_deg, det_position_m):
        #         up y^  /z  down stream
        #             | /
        #  door x <- -|
        # the base vector is in the lab frame now
        vector = np.diag(np.ones(3, dtype=np.float64))
        print(vector.shape)
        # apply detector rotation to base vectors
        for ax, angle in enumerate(det_rotate_deg):
            vector = rotate_vector(vector, ax, angle)  # lab frame

        # convert to sample frame, needed for reflection geometry
        origin = np.array(det_position_m)
        if self.sg_type == "reflection":
            # update incident angle
            self.incident_angle_deg = -sample_rotate_deg[0]
            for ax, angle in enumerate(sample_rotate_deg):
                vector = rotate_vector(vector, ax, -angle)  # sample frame
                origin = rotate_vector(origin, ax, -angle)
        # rot_matrix @ vector -> 3 columns are the new bases; transpose
        vector = vector.T
        return vector, origin

    def calculate_positions(self, oversampling=1):
        # construct 3D position arrays for the center of pixels
        shape = self.shape * oversampling
        vcg, hcg = np.meshgrid(
            np.arange(shape[0]) - self.beam_center[1] * oversampling,
            np.arange(shape[1]) - self.beam_center[0] * oversampling,
            indexing="ij",
        )

        pos = (
            vcg.reshape(*shape, 1) * self.vector[1]
            + hcg.reshape(*shape, 1) * self.vector[0]
        )

        pos *= -1.0 * self.pixel_size_m / oversampling

        # pos is now 3 x N x M array
        pos = np.ascontiguousarray(np.moveaxis(pos, 2, 0))

        # apply detector linear translate
        pos += self.origin.reshape(3, 1, 1)

        # dist is N x M distance
        dist = np.linalg.norm(pos, ord=2, axis=0)
        return (pos, dist), {"x": hcg, "y": vcg}

    def calculate_solid_angle(self, position, use_analytical=False):
        pos, dist = position
        if not use_analytical:
            # vectors' length is 1
            dot_product = np.abs(np.dot(self.vector[2], pos.reshape(3, -1)))
            cos_theta = dot_product / (dist.ravel())
            theta = np.arccos(cos_theta)
            solid_angle = np.pi / 2 - np.arccos(cos_theta)
            solid_angle = solid_angle.reshape(self.shape)
        else:
            raise NotImplementedError
            # shape = pos.shape
            # pos_corner = np.zeros((shape[0] + 1, shape[1] + 1, 3))
            # pos_corner[:-1, :-1] = pos
            # pos_corner[-1] = pos_corner[-2:] + self.vector[1]
            # pos_corner[:, -1] = pos_corner[:, -2] + self.vector[0]
            # pos_corner -= 0.5 * (self.vector[0] + self.vector[1])
        return {"solid_angle": solid_angle}

    def calculate_angles(self, pos_dist):
        pos, dist = pos_dist
        qmap = {}
        qmap["phi"] = np.arctan2(pos[1], pos[0])  # y / x
        if self.sg_type == "transmission":
            xy = np.hypot(pos[0], pos[1])
            qmap["tth"] = np.arcsin(xy / dist)  # z / dist
        else:  # reflection
            dist_xz = np.hypot(pos[0], pos[2])
            qmap["tth"] = np.arcsin(pos[0] / dist_xz)
            # z unit vector
            dist_yz = np.hypot(pos[1], pos[2])
            qmap["alpha"] = np.arcsin(pos[1] / dist_yz)
        return qmap

    def calculate_momentum_transfer(self, pos_dist):
        qmap = self.calculate_angles(pos_dist)

        pos, dist = pos_dist
        # h * c in kev = 12.398419840550368
        k0 = 2 * np.pi / (12.398419840550368 / self.energy_kev)
        k_in = k0 * self.vector[2].reshape(3, 1, 1)
        k_out = k0 * (pos / dist)  # it's a 3 x N x M array
        q_diff = k_out - k_in

        if self.sg_type == "transmission":
            # only keep qx and qy components
            qmap["q"] = np.hypot(q_diff[0], q_diff[1])
        elif self.sg_type == "reflection":
            qmap["q"] = np.linalg.norm(q_diff, ord=2, axis=0)
            # qx, qy, and qz follow the GIXS naming convention
            qmap["qy"] = q_diff[0]  # in plane; perpendicular to beam
            qmap["qz"] = q_diff[1]  # out of the plane
            qmap["qx"] = q_diff[2]  # along the beam
            qr = np.hypot(qmap["qx"], qmap["qy"])
            qr[qmap["qy"] < 0] *= -1
            qmap["qr"] = qr
            qmap["chi"] = np.arccos(q_diff[1] / qmap["q"])
        qmap["none"] = np.ones(self.shape, dtype=np.float64)

        return qmap

    def populate_qmap_unit(self, sg_type):
        if sg_type == "transmission":
            qmap_unit = TRANSMISSION_QMAP_UNIT.copy()
        elif sg_type == "reflection":
            qmap_unit = REFLECTION_QMAP_UNIT.copy()
        return qmap_unit

    def prepare_correction_matrix(self, correction_config):
        return
        correction_config["energy_keV"] = self.energy_kev
        cid = convert_to_hashable_str(correction_config)
        if self.current_correction_config != cid:
            self.current_correction_config = cid
            correction_mat = get_correction(
                self.position[0], self.qmap["solid_angle"], **correction_config
            )
            self.correction_mat = correction_mat
        return self.correction_mat

    def plot_all_maps(self):
        for k, v in self.qmap.items():
            plt.imshow(v, cmap=plt.cm.jet)
            plt.title(k)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"debug/{k}.png", dpi=300)
            plt.close()

    # def convert_data(self, data, method='pixel-pixel'):
    def get_remap_matrix(self, method="pixel-pixel", upscale=2):
        assert method in ["qy-qz", "qy-qx", "qr-qz", "tth-alpha_f"]
        map_names = method.split("-")
        if upscale == 1:
            qmap_all = self.qmap
        else:
            position, _ = self.calculate_positions(upscale)
            qmap_all = self.calculate_momentum_transfer(position)

        pm_sign = {"qx": 1, "qy": -1, "qr": -1, "qz": -1, "tth": -1, "alpha_f": -1}

        def remap_coordinates(map_name, upscale):
            qmap = qmap_all[map_name] * pm_sign[map_name]
            if map_name in ["qz", "qx", "alpha_f"]:
                line = qmap[:, int(self.beam_center[0] * upscale)]
            elif map_name in ["qy", "tth", "qr"]:
                # diff = self.qmap['alpha_f'][:,0] - self.incident_angle_deg
                diff = qmap_all["alpha_f"][:, 0]
                vidx = np.argmin(np.abs(diff))
                line = qmap[vidx]
            delta = np.abs(np.mean(np.diff(line))) * upscale
            vrange = (np.min(qmap), np.max(qmap))
            # vrange = (np.min(line), np.max(line))
            map_idx = (qmap.ravel() - vrange[0]) / delta
            map_idx = np.floor(map_idx + 0.5).astype(np.int64)
            size = np.max(map_idx) + 1
            return map_idx, size, vrange

        def construct_upscale_matrix(shape, upscale):
            size = shape[0] * shape[1]
            mat1 = np.arange(size).reshape(shape)
            block = np.ones((upscale, upscale), dtype=np.int64)
            mat2 = np.kron(mat1, block)
            col = mat2.ravel().astype(np.int64)
            row = np.arange(col.size)

            # normalize
            ones = np.ones_like(row) / (upscale * upscale)
            ups_mat = coo_matrix((ones, (row, col)))
            return ups_mat

        # construct convert matrix - SM->N
        map_idx0, size0, vrange0 = remap_coordinates(map_names[0], upscale)
        map_idx1, size1, vrange1 = remap_coordinates(map_names[1], upscale)
        row = map_idx0 + map_idx1 * size0
        out_shape = (size1, size0)
        out_size = size1 * size0
        extent = (*vrange0, *vrange1)

        in_shape = (self.shape * upscale).astype(np.int64)
        in_size = in_shape[0] * in_shape[1]
        cvt_ones = np.ones(in_size)
        col = np.arange(cvt_ones.size)

        cvt_mat = coo_matrix((cvt_ones, (row, col)), shape=(out_size, in_size))
        if upscale > 1:
            ups_mat = construct_upscale_matrix(self.shape, upscale)
            cvt_mat = cvt_mat @ ups_mat

        # compute normalization factor
        ones_input = np.ones(cvt_mat.shape[1])
        counts = cvt_mat @ ones_input
        nan_mask = counts.reshape(out_shape) == 0
        counts[counts == 0] = 1.0
        cvt_mat = diags(1.0 / counts) @ cvt_mat
        # convert to csr matrix for better performance
        cvt_mat = cvt_mat.tocsr()
        # check sparsity
        # sparsity = mat.getnnz() / np.prod(mat.shape)
        # print(sparsity)

        return cvt_mat, nan_mask, extent

    def remap_data(self, data, method):
        if method not in self.remap_solver:
            self.remap_solver[method] = self.get_remap_matrix(method)
        cvt_mat, nan_mask, extent = self.remap_solver[method]
        new_data = cvt_mat @ data.ravel()
        new_data = new_data.reshape(nan_mask.shape)
        new_data[nan_mask] = np.nan

        if False:
            plt.imshow(
                np.log10(new_data), cmap=plt.cm.jet, extent=extent, aspect="auto"
            )
            plt.title(method)
            plt.xlabel(map_names[0])
            plt.ylabel(map_names[1])
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"debug/{method}.png", dpi=600)
            plt.close()
        return new_data, extent

    def get_qmap_unit_range(self, map_name):
        xmap = self.qmap[map_name]
        return self.qmap_unit[map_name], (np.nanmin(xmap), np.nanmax(xmap))

    def get_linecuts(self, scat_img, lc_config):
        solver = self.get_linecut_solver(lc_config)
        mask, roi_mask, x_val, weights, counts = solver
        # print(mask.shape, roi_mask.shape, weights.shape, scat_img[mask].shape)
        y_val = np.bincount(roi_mask, weights=scat_img[mask], minlength=x_val.size + 1)[
            1:
        ]
        y_val *= weights
        masked_img = np.copy(scat_img)
        masked_img[mask == 0] = 1
        masked_img[masked_img == 0] = 1
        x_unit = self.qmap_unit[lc_config["plot_axis"][0]]
        result = {
            "x_val": x_val,
            "x_label": lc_config["plot_axis"][0] + f" ({x_unit})",
            "y_val": y_val,
            "y_label": "Average (a.u.)",
            "counts": counts,
            "masked_img": np.log10(masked_img),
            "mask": mask,
        }
        return result

    def get_linecut_solver(self, lc_config):
        mid = convert_to_hashable_str(lc_config)
        if not mid in self.lincut_solver_cache:
            solver = self.get_linecuts_map(lc_config)
            self.lincut_solver_cache[mid] = solver
        return self.lincut_solver_cache[mid]

    def get_linecuts_map(self, lc_config):
        selected_axis, style, number_points = lc_config["plot_axis"]
        constraints = lc_config["constraints"]

        mask = self.get_constraint_mask(constraints)
        x_all = self.qmap[selected_axis][mask]
        x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
        if style == "linear":
            x_pos = np.linspace(x_min, x_max, number_points + 1)
            x_val = (x_pos[:-1] + x_pos[1:]) / 2.0
        else:
            x_pos_log = np.logspace(np.log(x_min), np.log(x_max), number_points + 1)
            x_pos = np.exp(x_pos_log)
            x_val = np.sqrt(x_pos[:-1] * x_pos[1:])

        roi_mask = np.zeros_like(x_all, dtype=np.int32)
        roi_prev = x_all < x_pos[0]
        for n in range(1, number_points + 1):
            if n < number_points:
                roi_curr = x_all < x_pos[n]
            else:
                roi_curr = x_all <= x_pos[n]
            roi = (~roi_prev) * (roi_curr)
            roi_mask[roi] = n
            roi_prev = roi_curr

        counts = np.bincount(roi_mask, minlength=number_points)[1:]
        weights = 1.0 / np.clip(counts, a_min=1, a_max=None)
        return mask, roi_mask, x_val, weights, counts

    def get_constraint_mask(self, constraints):
        mask = np.ones(self.shape, dtype=bool)
        for n in range(len(constraints) // 4):
            map_name, vmin, vmax, logic = constraints[n * 4 : n * 4 + 4]
            if map_name == "none":
                continue
            mask_t = np.ones_like(mask)
            target = self.qmap[map_name] * self.mask
            if vmin is not None:
                lo_bound = target >= vmin
                mask_t = np.logical_and(lo_bound, mask_t)
            if vmax is not None:
                up_bound = target <= vmax
                mask_t = np.logical_and(up_bound, mask_t)

            if logic == "AND":
                mask = np.logical_and(mask, mask_t)
            elif logic == "OR":
                mask = np.logical_or(mask, mask_t)
            elif logic == "NOT":
                mask = np.logical_and(mask, ~mask_t)
            elif logic == "XOR":
                mask = np.logical_xor(mask, mask_t)
            else:
                raise ValueError(f"{logic} operation not supported.")
        return mask


def test_go_01():
    sg = ScatteringGeometry(
        (256, 512),  # shape
        (128, 100),  # beam center
        (0, 0, 5.0),  # distance
        det_rotate_deg=(0, 0, 0),
        sample_rotate_deg=(0, 0, 0),
        pixel_size_m=75e-6,
        sg_type="transmission",
    )
    # for k, v in sg.angles_rad.items():
    #     plt.imshow(v)
    #     plt.title(k)
    #     plt.colorbar()
    #     plt.show()
    #     plt.close()


def test_go_02():
    sg = ScatteringGeometry(
        (256, 512),  # shape
        (128, 100),  # beam center
        (0, 0, 3000),
        det_rotate_deg=(0, 0, 10),
        sample_rotate_deg=None,
        pixel_size_m=75e-6,
    )
    for k, v in sg.angles_rad.items():
        plt.imshow(v)
        plt.title(k)
        plt.colorbar()
        plt.show()
        plt.close()


def test_go_03():
    sg = ScatteringGeometry(
        (2048, 2560),  # shape
        (1024, 2000),  # beam center
        (0, 0, 5000),
        energy_kev=8.0,
        det_rotate_deg=(0, 0, 0),
        sample_rotate_deg=(-1.0, 0, 0),
        pixel_size_m=75e-6,
    )
    print(sg.sg_type)
    # for k, v in sg.angles_rad.items():
    # for k, v in zip(list('xyz'), sg.pos):
    #     plt.imshow(v)
    #     plt.title(k)
    #     plt.colorbar()
    #     plt.show()
    #     plt.close()
    # return
    # for k, v in sg.angles_rad.items():
    for k, v in sg.q_info.items():
        plt.imshow(v)
        plt.title(k)
        plt.colorbar()
        plt.show()
        plt.close()


def test_go_petra():
    # 'alpha_incident_deg': 0.60,
    # 'det_distance': 5.00,
    # 'beam_center': (1207, 2523),
    # 'det_pixel_size': 75e-6,
    # 'det_y': 0,
    # 'det_z': 0,
    # 'energy': 8.00,
    # 'ring_current': 101
    sg = ScatteringGeometry(
        (2070, 2167),  # shape
        (1207, 2523),  # beam center
        (0, 0, 5000),
        energy_kev=8.0,
        det_rotate_deg=(0, 0, 0),
        sample_rotate_deg=(-0.6, 0, 0),
        pixel_size_mm=75e-3,
    )

    # sg.plot_all_maps()
    data = skio.imread(
        "../../tests/data/cssi_petra/LargeYOLraw_data_060_scatterings32.tif"
    )

    t0 = time.perf_counter()
    sg.remap_data(data, "qy-qz")
    t1 = time.perf_counter()
    print(t1 - t0)

    t0 = time.perf_counter()
    sg.remap_data(data, "qy-qz")
    t1 = time.perf_counter()
    print(t1 - t0)


if __name__ == "__main__":
    # test_go_petra()
    test_go_01()
