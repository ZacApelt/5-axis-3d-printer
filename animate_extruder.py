"""Animate CSV poses x,y,z,nx,ny,nz using Open3D.

Install: python -m pip install numpy open3d
Run:     python animate_extruder.py layer.csv --speed 10
         python animate_extruder.py layer.csv --post --scene printed.stl

Normals point FROM TIP TOWARD BODY (same convention as the slicer).
Units: millimetres. Geometry defaults match the notebook: cone half-angle
25 degrees, diameter 30, total cone+cylinder length 100. Optional post:
width 20, length 150, pivot 75 from tip, always vertical in world coordinates.

Controls: SPACE pause/resume; R restart; UP/DOWN playback speed;
LEFT/RIGHT seek 1 simulated second; N/B next/previous CSV pose (pauses);
W toggle tool wireframe; H hide/show tool; mouse drag/scroll orbit/zoom;
Q or ESC close. Starts paused so you can position the camera.

Grey = entire planned polyline, green = traversed segments, yellow = current
partial segment, red sphere = nozzle tip. This is a visualisation, not a
collision detector. Every CSV row connects to the next; these six columns
cannot encode separate paths or distinguish extrusion from travel.

No feedrates are provided in the CSV: timing uses --speed (mm/s) and
--angular-speed (degrees/s). Directions interpolate on the unit sphere;
this need not reproduce the printer controller's rotary-axis interpolation.
Exact opposite normals have no unique interpolation axis; a consistent
arbitrary perpendicular direction is used for that exceptional case.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np


def load_poses(filename):
    with open(filename, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV is empty.')
        reader.fieldnames = [s.strip().lower() for s in reader.fieldnames]
        required = ('x', 'y', 'z', 'nx', 'ny', 'nz')
        if not set(required).issubset(reader.fieldnames):
            raise ValueError('CSV requires headings x,y,z,nx,ny,nz.')
        rows = []
        for row in reader:
            try:
                rows.append([float(row[name]) for name in required])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid numeric value at CSV line {reader.line_num}.') from exc
    poses = np.asarray(rows, dtype=float).reshape(-1, 6)
    if not len(poses) or not np.isfinite(poses).all():
        raise ValueError('CSV must contain finite poses.')
    lengths = np.linalg.norm(poses[:, 3:], axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('CSV contains a zero normal.')
    poses[:, 3:] /= lengths[:, None]
    return poses


def interpolate_normal(a, b, fraction):
    dot = np.clip(a @ b, -1., 1.)
    if dot > 0.999999:
        n = (1-fraction)*a + fraction*b
    else:
        perpendicular = b-dot*a
        length = np.linalg.norm(perpendicular)
        if length < 1e-10:
            axis = np.eye(3)[np.argmin(np.abs(a))]
            perpendicular = axis-(axis @ a)*a
            length = np.linalg.norm(perpendicular)
        perpendicular /= length
        angle = np.arccos(dot)*fraction
        n = np.cos(angle)*a + np.sin(angle)*perpendicular
    return n / np.linalg.norm(n)


def tool_rotation(normal):
    """Map local +Z (nozzle direction) to -normal; body extends along -Z."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    theta = np.arctan2(-n[0], n[1]) if np.hypot(n[0], n[1]) > 1e-12 else 0.
    local_x = np.array([np.cos(theta), np.sin(theta), 0.])
    local_z = -n
    local_y = np.cross(local_z, local_x)
    return np.column_stack((local_x, local_y, local_z))


class Trajectory:
    def __init__(self, poses, speed, angular_speed):
        self.poses = poses
        linear = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
        angular = np.arccos(np.clip(np.sum(poses[:-1, 3:]*poses[1:, 3:], axis=1), -1., 1.))
        durations = np.maximum(linear/speed, angular/np.deg2rad(angular_speed))
        self.times = np.r_[0., np.cumsum(durations)]
        self.duration = float(self.times[-1])

    def sample(self, elapsed):
        if len(self.poses) == 1 or elapsed >= self.duration:
            return self.poses[-1, :3].copy(), self.poses[-1, 3:].copy(), len(self.poses)-1
        i = max(0, min(int(np.searchsorted(self.times, max(elapsed, 0), side='right'))-1, len(self.poses)-2))
        dt = self.times[i+1]-self.times[i]
        t = np.clip((elapsed-self.times[i])/dt, 0., 1.) if dt > 0 else 0.
        position = (1-t)*self.poses[i, :3] + t*self.poses[i+1, :3]
        normal = interpolate_normal(self.poses[i, 3:], self.poses[i+1, 3:], t)
        return position, normal, i


def animate(args):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('Install Open3D in this Python environment: python -m pip install open3d') from exc
    poses = load_poses(args.csv)
    if args.flip_normals:
        poses[:, 3:] *= -1
    trajectory = Trajectory(poses, args.speed, args.angular_speed)
    radius = args.diameter / 2
    cone_height = radius / np.tan(np.deg2rad(args.cone_angle))
    cylinder_height = args.length-cone_height
    if cylinder_height <= 0:
        raise ValueError('--length must exceed the calculated cone height.')

    # Construct once. Animate vertex transforms, without rebuilding meshes.
    cone = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=cone_height, resolution=30)
    cone.translate((0, 0, -cone_height))
    cone.paint_uniform_color([0.90, 0.59, 0.18])
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=cylinder_height, resolution=30)
    cylinder.translate((0, 0, -cone_height-cylinder_height/2))
    cylinder.paint_uniform_color([0.35, 0.53, 0.75])
    tool = cone+cylinder
    tool.compute_vertex_normals()
    tool_vertices = np.asarray(tool.vertices).copy()
    tool_normals = np.asarray(tool.vertex_normals).copy()
    tool_face_normals = np.asarray(tool.triangle_normals).copy()

    tip = o3d.geometry.TriangleMesh.create_sphere(radius=args.tip_radius, resolution=8)
    tip.paint_uniform_color([1., 0.12, 0.12])
    tip.compute_vertex_normals()
    tip_vertices = np.asarray(tip.vertices).copy()
    post = None
    if args.post:
        post = o3d.geometry.TriangleMesh.create_box(width=20., height=20., depth=150.)
        post.translate((-10., -10., 0.))
        post.paint_uniform_color([0.55, 0.57, 0.60])
        post.compute_vertex_normals()
        post_vertices = np.asarray(post.vertices).copy()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window('Extruder playback - SPACE play/pause | R restart | arrows speed/seek', 1280, 850):
        raise RuntimeError('Could not create Open3D window. Run from a local desktop terminal.')
    try:
        path = o3d.geometry.LineSet()
        path.points = o3d.utility.Vector3dVector(poses[:, :3])
        path.lines = o3d.utility.Vector2iVector(np.column_stack((np.arange(len(poses)-1), np.arange(1,len(poses)))))
        colors = np.tile([0.42, 0.44, 0.48], (len(poses)-1, 1))
        path.colors = o3d.utility.Vector3dVector(colors)
        if len(poses)>1:
            vis.add_geometry(path)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.)
        vis.add_geometry(axes)
        if args.scene:
            scene = o3d.io.read_triangle_mesh(str(args.scene))
            if scene.is_empty():
                raise ValueError('Scene mesh is empty or could not be read.')
            scene.compute_vertex_normals()
            scene.paint_uniform_color([0.60, 0.62, 0.65])
            vis.add_geometry(scene)
        if args.plate_size:
            plate = o3d.geometry.TriangleMesh.create_box(width=args.plate_size, height=args.plate_size, depth=1.)
            plate.translate((-args.plate_size/2, -args.plate_size/2, -1.))
            plate.paint_uniform_color([0.20, 0.23, 0.27])
            plate.compute_vertex_normals()
            vis.add_geometry(plate)

        def move_tool(position, normal):
            R = tool_rotation(normal)
            np.asarray(tool.vertices)[:] = tool_vertices @ R.T + position
            np.asarray(tool.vertex_normals)[:] = tool_normals @ R.T
            np.asarray(tool.triangle_normals)[:] = tool_face_normals @ R.T
            np.asarray(tip.vertices)[:] = tip_vertices + position
            if post is not None:
                np.asarray(post.vertices)[:] = post_vertices + position + 75.*normal

        move_tool(poses[0,:3], poses[0,3:])
        vis.add_geometry(tool)
        vis.add_geometry(tip, reset_bounding_box=False)
        if post is not None:
            vis.add_geometry(post)
        partial = o3d.geometry.LineSet()
        partial.points = o3d.utility.Vector3dVector(np.tile(poses[0,:3], (2,1)))
        partial.lines = o3d.utility.Vector2iVector([[0,1]])
        partial.colors = o3d.utility.Vector3dVector([[1.,.85,.1]])
        vis.add_geometry(partial, reset_bounding_box=False)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.065,0.075,0.09])
        opt.line_width = 2.
        opt.mesh_show_back_face = True
        view = vis.get_view_control()
        view.set_up([0.,0.,1.])
        view.set_front([1.,-1.,0.7])
        view.set_lookat(poses[:,:3].mean(axis=0))
        view.set_zoom(.65)
        state = {'elapsed':0., 'playing':args.autoplay, 'rate':1., 'visible':True}

        def pause(v):
            if state['elapsed'] >= trajectory.duration:
                state['elapsed'] = 0.
            state['playing'] = not state['playing']
            return False
        def restart(v):
            state['elapsed'] = 0.
            return False
        def change_speed(factor):
            def callback(v):
                state['rate'] = float(np.clip(state['rate']*factor, 1/64, 64))
                print(f"Playback: {state['rate']:g}x", flush=True)
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
                side = 'right' if direction>0 else 'left'
                i = int(np.searchsorted(trajectory.times,state['elapsed'],side=side))
                if direction<0: i-=1
                state['elapsed'] = float(trajectory.times[np.clip(i,0,len(poses)-1)])
                return False
            return callback
        def wireframe(v):
            opt.mesh_show_wireframe = not opt.mesh_show_wireframe
            return False
        def hide(v):
            state['visible'] = not state['visible']
            for g in ([tool] if post is None else [tool,post]):
                if state['visible']: v.add_geometry(g, reset_bounding_box=False)
                else: v.remove_geometry(g, reset_bounding_box=False)
            return False
        for key, callback in [(32,pause),(ord('R'),restart),(265,change_speed(2)),
                              (264,change_speed(.5)),(262,seek(1)),(263,seek(-1)),
                              (ord('N'),step(1)),(ord('B'),step(-1)),
                              (ord('W'),wireframe),(ord('H'),hide)]:
            vis.register_key_callback(key,callback)
        print(f'{len(poses):,} poses; duration {trajectory.duration:.1f}s at {args.speed:g} mm/s.')
        print('SPACE play/pause | R restart | UP/DOWN speed | LEFT/RIGHT seek | N/B step | W wireframe | H hide tool')
        print('Grey: planned. Green: completed. Yellow: current segment. CSV row order is followed exactly.')
        previous = time.perf_counter()
        colored_to = 0
        while vis.poll_events():
            frame_start = time.perf_counter()
            delta = frame_start-previous
            previous = frame_start
            if state['playing']:
                state['elapsed'] += delta*state['rate']
                if state['elapsed'] >= trajectory.duration:
                    if args.loop and trajectory.duration>0:
                        state['elapsed'] %= trajectory.duration
                    else:
                        state['elapsed'] = trajectory.duration
                        state['playing'] = False
            position, normal, index = trajectory.sample(state['elapsed'])
            move_tool(position,normal)
            if state['visible']:
                vis.update_geometry(tool)
                if post is not None: vis.update_geometry(post)
            vis.update_geometry(tip)
            if index != colored_to:
                if index < colored_to:
                    colors[index:] = [.42,.44,.48]
                else:
                    colors[colored_to:index] = [.15,.95,.45]
                np.asarray(path.colors)[:] = colors
                vis.update_geometry(path)
                colored_to = index
            np.asarray(partial.points)[:] = np.vstack((poses[index,:3],position))
            vis.update_geometry(partial)
            vis.update_renderer()
            time.sleep(max(0., 1/args.fps-(time.perf_counter()-frame_start)))
    finally:
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, help='CSV containing x,y,z,nx,ny,nz')
    parser.add_argument('--speed', type=float, default=10., help='Linear speed in mm/s (default 10)')
    parser.add_argument('--angular-speed', type=float, default=90., help='Maximum interpolated angular speed in degrees/s')
    parser.add_argument('--diameter', type=float, default=30.)
    parser.add_argument('--length', type=float, default=100., help='Cone plus cylinder length')
    parser.add_argument('--cone-angle', type=float, default=25., help='Cone half-angle in degrees')
    parser.add_argument('--tip-radius', type=float, default=.25, help='Visual tip marker radius')
    parser.add_argument('--post', action='store_true', help='Include the world-vertical support post')
    parser.add_argument('--scene', type=Path, help='Optional mesh, e.g. STL of previously printed material')
    parser.add_argument('--plate-size', type=float, default=50., help='Plate side length centred at XY=0; zero disables')
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--autoplay', action='store_true')
    parser.add_argument('--loop', action='store_true', help='Replay; resets from final pose to first without a travel move')
    parser.add_argument('--flip-normals', action='store_true', help='Only use if CSV normals point toward material rather than tool body')
    args = parser.parse_args()
    for name in ('speed','angular_speed','diameter','length','tip_radius','fps'):
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<=0:
            parser.error(f'{name} must be positive and finite.')
    if not 0<args.cone_angle<90 or not np.isfinite(args.plate_size) or args.plate_size<0:
        parser.error('Require 0 < cone-angle < 90 and plate-size >= 0.')
    try:
        animate(args)
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__ == '__main__':
    main()
