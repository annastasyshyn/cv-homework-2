import cv2
import numpy as np
import numpy.typing as npt
import nvdiffrast.torch as dr
import torch


class MeshWarper:
    """This class implements the interpolation of image by defining the forward transformation of regular grid"""

    def __init__(self):
        if not torch.cuda.is_available():
            raise Exception("CUDA device is not available")
        else:
            print("CUDA device available. Using cuda:0 for warping.")
        self.device = "cuda:0"
        self.glctx = dr.RasterizeCudaContext(device=self.device)

    @staticmethod
    def _build_mvp_cam_matrix(R, T, f, pp, image_shape, near=None, far=None):
        """Return camera pose (translation) and model-view-projection matrix used by opengl"""
        B, _ = T.shape
        world2cam = torch.cat([
            torch.cat([
                R,
                R.new_zeros(B, 1, 3)
            ], dim=1),
            torch.nn.functional.pad(T, pad=(0, 1), mode='constant', value=1.0)[:, :, None],
        ], dim=2)

        flip_yz = T.new_tensor([[
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ]]).expand(B, -1, -1)

        # calculate projection matrix from perspective camera params and near/far clipping planes
        H, W = image_shape
        # if near and far are None estimate them by +- 1 around zero
        if near is None:
            near = max(-5 + T[:, 2].min(), 0.01)
        if far is None:
            far = 5 + T[:, 2].max()

        canvas_x, canvas_y = near * W / f[:, 0], near * H / f[:, 1]
        cx, cy = pp[:, 0], pp[:, 1]

        top = canvas_y * cy / H
        bottom = -canvas_y * (1 - cy / H)
        left = -canvas_x * cx / W
        right = canvas_x * (1 - cx / W)

        zero_t, one_t = T.new_tensor([0]).expand(B), T.new_tensor([1]).expand(B)
        far_t, near_t = T.new_tensor([far]).expand(B), T.new_tensor([near]).expand(B)

        proj4x4 = torch.stack([
            torch.stack([2 * near_t / (right - left), zero_t, (right + left) / (right - left), zero_t], dim=1),
            torch.stack([zero_t, 2 * near_t / (top - bottom), (top + bottom) / (top - bottom), zero_t], dim=1),
            torch.stack([zero_t, zero_t, -(far_t + near_t) / (far_t - near_t), -2 * far_t * near_t / (far_t - near_t)],
                        dim=1),
            torch.stack([zero_t, zero_t, -one_t, zero_t], dim=1)
        ], dim=1)

        mv = torch.bmm(flip_yz, world2cam)
        mvp = torch.bmm(proj4x4, mv)

        return mvp

    def warp_grid(
            self,
            image: npt.NDArray[np.uint8],
            grid_src: npt.NDArray[np.float64],
            grid_dst: npt.NDArray[np.float64],
            faces: npt.NDArray[np.int64],
            output_shape: tuple[int, int]) -> npt.NDArray[np.uint8]:
        """
        Warp `image` according to the transformation from `grid_src` to `grid_dst` points.
        Image is represented as 2D triangular mesh, whose faces are defined in `faces`.
        Args:
            image (npt.NDArray[np.uint8]): input image of shape (H, W) if gray or (H, W, C) if it has multiple channels
            grid_src (npt.NDArray[np.float64]): x- and y- coordinates of input grid over the image of shape (N, 2)
            grid_dst: (npt.NDArray[np.float64]): x- and y- coordinates of warped grid of shape (N, 2)
            faces: (npt.NDArray[np.int64]): indices of triangular faces that define input mesh of shape (M, 3)
            output_shape (tuple[int, int]): shape of the output image represented as (W, H)

        Returns:
            (npt.NDArray[np.uint8]): warped image of same shape as input `image`
        """
        W, H = output_shape

        # convert inputs from numpy to torch tensors
        if len(image.shape) == 2:
            image = image[:, :, None]
        image = torch.from_numpy(image).to(device=self.device, dtype=torch.float32) / 255.
        grid_src = torch.tensor(grid_src, device=self.device, dtype=torch.float32)
        grid_dst = torch.tensor(grid_dst, device=self.device, dtype=torch.float32)
        faces = torch.tensor(faces, device=self.device, dtype=torch.int32)

        # build camera matrix at zero
        mvp = self._build_mvp_cam_matrix(
            R=torch.eye(3, device=self.device)[None],
            T=torch.zeros(3, device=self.device)[None],
            f=torch.tensor([1, 1], device=self.device)[None],
            pp=torch.tensor([0., 0.], device=self.device)[None],
            image_shape=(H, W)
        )

        # place meshgrid on the plane Z=1
        verts = torch.cat([grid_dst, grid_dst.new_ones(grid_dst.shape[0], 1)], dim=1)

        # transform verts to clip-space
        verts_clip = torch.matmul(
            torch.nn.functional.pad(verts, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(mvp, 1, 2))  # [B, N, 4]

        # rasterize deformed meshgrid
        rast, rast_db = dr.rasterize(self.glctx, verts_clip, faces, (H, W), grad_db=False)
        texture_uv, texture_uv_db = dr.interpolate(grid_src / grid_src.new_tensor([W, H]), rast, faces, rast_db, "all")
        output = dr.texture(image[None], texture_uv, texture_uv_db, filter_mode="linear-mipmap-linear",
                            boundary_mode="zero", max_mip_level=8)
        output = dr.antialias(output, rast, verts_clip, faces)
        output = output.flip((1,)).contiguous()[0]

        return (output * 255.).squeeze(2).byte().cpu().numpy()

    @staticmethod
    def build_meshgrid(
            img_width: int,
            img_height: int,
            grid_rows: int,
            grid_cols: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        Build triangular 2D mesh over the image in regular grid pattern.
        Args:
            img_width: Horizontal size of the image in pixels
            img_height: Vertical size of the image in pixels
            grid_rows: Number of rows in the resulting meshgrid. Actual number of points is (grid_rows+1)
            grid_cols: Number of cols in the resulting meshgrid. Actual number of points is (grid_cols+1)

        Returns:
            (npt.NDArray[np.float64]): 2d vertices defining the grid of shape (N, 2)
            (npt.NDArray[np.int64]): indices of faces defining triangular meshgrid of shape (M, 3)
        """
        vertices = np.stack(
            np.meshgrid(
                np.linspace(0, img_width, num=grid_cols + 1),
                np.linspace(0, img_height, num=grid_rows + 1)
            ),
            axis=-1
        ).reshape(-1, 2)

        # build faces
        r = np.arange(grid_rows)
        c = np.arange(grid_cols)
        rr, cc = np.meshgrid(r, c, indexing="ij")

        v00 = rr * (grid_cols + 1) + cc
        v01 = v00 + 1
        v10 = v00 + (grid_cols + 1)
        v11 = v10 + 1

        tri1 = np.stack([v00, v10, v11], axis=-1)  # CCW
        tri2 = np.stack([v00, v11, v01], axis=-1)  # CCW
        faces = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0).astype(np.int64)
        return vertices, faces

    @staticmethod
    def draw_meshgrid(verts: npt.NDArray[np.float64], faces: npt.NDArray[np.int64], W: int, H: int):
        """
        Draw 2d triangular meshgrid as an image.
        Args:
            verts (npt.NDArray[np.float64]): 2d vertices defining the grid of shape (N, 2)
            faces (npt.NDArray[np.int64]): indices of faces defining triangular meshgrid of shape (M, 3)
            W: Horizontal size of the image in pixels
            H: Vertical size of the image in pixels

        Returns:
            (npt.NDArray[np.uint8]): drawing of the grid
        """
        # Create blank image
        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Draw mesh with OpenCV
        for f in faces:
            pts = verts[f].astype(np.int32)  # (3,2)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
        return img
