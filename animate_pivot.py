"""Animate pivot CSV: x,y,z,theta,phi (mm, radians by default).

Install: python -m pip install numpy open3d
Run: python animate_pivot.py layer.csv
Use --degrees if CSV angles are degrees.

RGB pivot frame translates with XYZ and rotates ONLY about world Z by theta.
Magenta arrow points from the pivot toward the nozzle tip (52 mm).
Tip = pivot + length*(sin(theta)*cos(phi), -cos(theta)*cos(phi), -sin(phi)).
The plotted polyline connects reconstructed tip waypoints, not pivot positions.
Use --reverse-vector to reverse only the displayed arrow (not the tip path).
The CSV position is already the pivot: no additional IK offset is applied.
Angles interpolate literally, NOT wrapped: explicit unwinds are preserved.
Timing assumes constant translation/axis speeds, without acceleration limits.
Every row connects to the next; no extrusion/travel distinction is available.

SPACE play/pause, R restart, UP/DOWN speed, LEFT/RIGHT seek one second,
N/B step CSV pose, mouse orbit/zoom, Q/ESC close. Starts paused.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np


def load_poses(filename, degrees=False):
    with open(filename, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [s.strip().lower() for s in (reader.fieldnames or [])]
        names = ('x', 'y', 'z', 'theta', 'phi')
        if not set(names).issubset(reader.fieldnames):
            raise ValueError('CSV requires x,y,z,theta,phi headings.')
        rows = []
        for row in reader:
            try:
                rows.append([float(row[k]) for k in names])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid value at CSV line {reader.line_num}.') from exc
    poses = np.asarray(rows, dtype=float).reshape(-1, 5)
    if not len(poses) or not np.isfinite(poses).all():
        raise ValueError('CSV must contain finite poses.')
    if degrees:
        poses[:, 3:] = np.deg2rad(poses[:, 3:])
    return poses


def yaw_frame(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def normal_frame(theta, phi):
    """Right-handed rotation with local +Z along the slicer normal."""
    x = np.array([np.cos(theta), np.sin(theta), 0.])
    n = np.array([-np.sin(theta)*np.cos(phi),
                  np.cos(theta)*np.cos(phi), np.sin(phi)])
    return np.column_stack((x, np.cross(n, x), n))


def tip_position(pose, length=52.):
    """CSV XYZ is pivot; slicer normal points from tip toward body."""
    return pose[:3] - length * normal_frame(pose[3], pose[4])[:, 2]


class Trajectory:
    def __init__(self, poses, speed=10., angular_speed=90.):
        self.poses = poses
        linear = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
        # Each rotary axis is limited separately; retain full unwinding angles.
        angular = np.max(np.abs(np.diff(poses[:, 3:], axis=0)), axis=1)
        durations = np.maximum(linear/speed, angular/np.deg2rad(angular_speed))
        self.times = np.r_[0., np.cumsum(durations)]
        self.duration = float(self.times[-1])

    def sample(self, elapsed):
        if len(self.poses) == 1 or elapsed >= self.duration:
            return self.poses[-1].copy(), len(self.poses)-1
        i = int(np.searchsorted(self.times, max(0., elapsed), side='right'))-1
        i = int(np.clip(i, 0, len(self.poses)-2))
        dt = self.times[i+1]-self.times[i]
        f = np.clip((elapsed-self.times[i])/dt, 0., 1.) if dt else 0.
        return (1-f)*self.poses[i]+f*self.poses[i+1], i


def animate(args):
    import open3d as o3d
    poses = load_poses(args.csv, args.degrees)
    trajectory = Trajectory(poses, args.speed, args.angular_speed)
    tips = np.array([tip_position(p, args.vector_length) for p in poses])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
    head = min(5., args.vector_length*.2)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=args.vector_length*.006,
        cone_radius=args.vector_length*.018,
        cylinder_height=args.vector_length-head, cone_height=head)
    arrow.paint_uniform_color([1., .15, .75])
    arrow.compute_vertex_normals()
    originals = [(g, np.asarray(g.vertices).copy(),
                  np.asarray(g.vertex_normals).copy(),
                  np.asarray(g.triangle_normals).copy()) for g in (frame, arrow)]

    def move(pose):
        rotations = [yaw_frame(pose[3]), normal_frame(pose[3], pose[4])]
        if not args.reverse_vector:
            rotations[1] = rotations[1] @ np.diag([1., -1., -1.])
        for (g, vertices, normals, faces), rotation in zip(originals, rotations):
            np.asarray(g.vertices)[:] = vertices @ rotation.T + pose[:3]
            np.asarray(g.vertex_normals)[:] = normals @ rotation.T
            np.asarray(g.triangle_normals)[:] = faces @ rotation.T

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window('Pivot playback | SPACE play/pause | arrows speed/seek', 1280, 850):
        raise RuntimeError('Could not open window; run on a local desktop.')
    try:
        path = o3d.geometry.LineSet()
        path.points = o3d.utility.Vector3dVector(tips)
        path.lines = o3d.utility.Vector2iVector(
            np.column_stack((np.arange(len(poses)-1), np.arange(1, len(poses)))))
        colors = np.tile([.45, .47, .50], (len(poses)-1, 1))
        path.colors = o3d.utility.Vector3dVector(colors)
        if len(poses)>1:
            vis.add_geometry(path)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size*.5))
        if args.scene:
            scene = o3d.io.read_triangle_mesh(str(args.scene))
            if scene.is_empty():
                raise ValueError('Scene mesh could not be read.')
            scene.compute_vertex_normals()
            scene.paint_uniform_color([.55, .58, .62])
            vis.add_geometry(scene)
        move(poses[0])
        vis.add_geometry(frame)
        vis.add_geometry(arrow)
        partial = o3d.geometry.LineSet()
        partial.points = o3d.utility.Vector3dVector(np.tile(tips[0], (2, 1)))
        partial.lines = o3d.utility.Vector2iVector([[0, 1]])
        partial.colors = o3d.utility.Vector3dVector([[1., .85, .1]])
        vis.add_geometry(partial, reset_bounding_box=False)
        opt = vis.get_render_option()
        opt.background_color = np.array([.065, .075, .09])
        opt.mesh_show_back_face = True
        view = vis.get_view_control()
        view.set_up([0., 0., 1.])
        view.set_front([1., -1., .7])
        view.set_lookat(tips.mean(axis=0))
        view.set_zoom(.65)
        state = dict(elapsed=0., playing=args.autoplay, rate=1.)

        def pause(v):
            if state['elapsed'] >= trajectory.duration:
                state['elapsed'] = 0.
            state['playing'] = not state['playing']
            return False

        def restart(v):
            state['elapsed'] = 0.
            return False

        def speed(factor):
            def callback(v):
                state['rate'] = float(np.clip(state['rate']*factor, 1/64, 64))
                print(f"Playback {state['rate']:g}x")
                return False
            return callback

        def seek(delta):
            def callback(v):
                state['elapsed'] = float(np.clip(state['elapsed']+delta, 0, trajectory.duration))
                return False
            return callback

        def step(direction):
            def callback(v):
                state['playing'] = False
                i = np.searchsorted(trajectory.times, state['elapsed'],
                                    side='right' if direction>0 else 'left')
                if direction<0:
                    i -= 1
                i = int(np.clip(i, 0, len(poses)-1))
                state['elapsed'] = float(trajectory.times[i])
                print(f'Row {i+1}: XYZ={poses[i,:3]}, theta/phi degrees={np.rad2deg(poses[i,3:])}')
                return False
            return callback

        for key, callback in [(32, pause), (ord('R'), restart),
                              (265, speed(2)), (264, speed(.5)),
                              (262, seek(1)), (263, seek(-1)),
                              (ord('N'), step(1)), (ord('B'), step(-1))]:
            vis.register_key_callback(key, callback)
        print(f'{len(poses):,} poses; {trajectory.duration:.1f} simulated seconds.')
        print('SPACE play/pause | R restart | UP/DOWN speed | LEFT/RIGHT seek | N/B step')
        print('RGB = yaw-only pivot frame; magenta = pivot-to-tip vector; polyline = tip waypoints. Literal rotary interpolation.')
        previous = time.perf_counter()
        colored_to = 0
        while vis.poll_events():
            now = time.perf_counter()
            dt, previous = now-previous, now
            if state['playing']:
                state['elapsed'] = min(trajectory.duration, state['elapsed']+dt*state['rate'])
                if state['elapsed'] >= trajectory.duration:
                    state['playing'] = False
            pose, index = trajectory.sample(state['elapsed'])
            move(pose)
            vis.update_geometry(frame)
            vis.update_geometry(arrow)
            if index != colored_to and len(poses)>1:
                if index < colored_to:
                    colors[index:] = [.45, .47, .50]
                else:
                    colors[colored_to:index] = [.15, .95, .45]
                np.asarray(path.colors)[:] = colors
                vis.update_geometry(path)
                colored_to = index
            np.asarray(partial.points)[:] = [tips[index], tip_position(pose, args.vector_length)]
            vis.update_geometry(partial)
            vis.update_renderer()
            time.sleep(max(0., 1/args.fps-(time.perf_counter()-now)))
    finally:
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--degrees', action='store_true', help='CSV angles are degrees (default radians)')
    parser.add_argument('--reverse-vector', action='store_true', help='Reverse displayed arrow only; tip reconstruction stays unchanged')
    parser.add_argument('--vector-length', type=float, default=52.)
    parser.add_argument('--axis-size', type=float, default=10.)
    parser.add_argument('--speed', type=float, default=10., help='mm/s')
    parser.add_argument('--angular-speed', type=float, default=90., help='degrees/s per rotary axis')
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--scene', type=Path, help='Optional static STL in the same world frame')
    parser.add_argument('--autoplay', action='store_true')
    args = parser.parse_args()
    for name in ('vector_length', 'axis_size', 'speed', 'angular_speed', 'fps'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name)<=0:
            parser.error(f'{name} must be positive and finite.')
    try:
        animate(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\nInstall dependencies: python -m pip install numpy open3d\n')


if __name__ == '__main__':
    main()
