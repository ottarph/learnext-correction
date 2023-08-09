# https://github.com/MiroK/gmshnics
from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
import numpy as np
import gmsh


def create_mesh():
    '''Mesh and entity function for the fluid & solid subdomain'''
    gmsh.initialize()
    # resolution
    resolution = 0.025  #0.05 #1 # 0.005 #0.1

    # geometric properties
    L = 2.5 #2.5 #20            # length of channel
    H = 0.41 #0.4 #6           # heigth of channel
    c = [0.2, 0.2, 0]  #[0.2, 0.2, 0] #[10, 3, 0]  # position of object
    r = 0.05 #0.05 #0.5 # radius of object

    # labels
    inflow = 1
    outflow = 2
    walls = 3
    noslipobstacle = 4
    obstacle = 5
    interface = 6
    fluid = 7
    solid = 8
    flag_tip = 9


    params = {"inflow" : inflow,
              "outflow": outflow,
              "noslip": walls,
              "obstacle_solid": noslipobstacle,
              "obstacle_fluid": obstacle,
              "interface": interface,
              "mesh_parts": True,
              "fluid": fluid,
              "solid": solid
              }

    vol = L*H
    vol_D_minus_obs = vol - np.pi*r*r
    geom_prop = {"barycenter_hold_all_domain": [0.5*L, 0.5*H],
                 "volume_hold_all_domain": vol,
                 "volume_D_minus_obstacle": vol_D_minus_obs,
                 "volume_obstacle": np.pi*r*r,
                 "length_pipe": L,
                 "heigth_pipe": H,
                 "barycenter_obstacle": [c[0], c[1]],
                 }

    model = gmsh.model
    fac = model.occ

    # Add circle
    pc = fac.addPoint(*c)
    sin = 0.5 # sin(30°)
    cos = np.sqrt(3)/2 # cos(30°)
    pc0 = fac.addPoint(*c)
    pc1 = fac.addPoint(c[0]-r, c[1], 0, 0.2*resolution)
    pc2 = fac.addPoint(0.24898979485, 0.21, 0, 0.2*resolution)
    pc3 = fac.addPoint(0.24898979485, 0.19,0, 0.2*resolution)
    circle1 = fac.addCircleArc(pc2, pc0, pc1)
    circle2 = fac.addCircleArc(pc1, pc0, pc3)
    circle3 = fac.addCircleArc(pc2, pc0, pc3)

    # Add elastic flag
    pf1 = fac.addPoint(0.6, 0.21, 0, 0.2*resolution)
    pf2 = fac.addPoint(0.6, 0.19, 0, 0.2*resolution)
    fl1 = fac.addLine(pc3, pf2)
    fl2 = fac.addLine(pf2, pf1)
    fl3 = fac.addLine(pf1, pc2)

    # obstacle
    obstacle_ = fac.addCurveLoop([fl1, fl2, fl3, circle1, circle2])
    flag = fac.addCurveLoop([circle3, fl1, fl2, fl3])

    # Add points with finer resolution on left side
    points = [fac.addPoint(0, 0, 0, resolution),
              fac.addPoint(L, 0, 0, resolution), #5*resolution
              fac.addPoint(L, H, 0, resolution), #5*resolution
              fac.addPoint(0, H, 0, resolution)]

    # Add lines between all points creating the rectangle
    channel_lines = [fac.addLine(points[i], points[i+1])
                     for i in range(-1, len(points)-1)]

    # Create a line loop and plane surface for meshing
    channel_loop = fac.addCurveLoop(channel_lines)
    plane_surface = fac.addPlaneSurface([channel_loop, obstacle_])
    plane_surface2 = fac.addPlaneSurface([flag])

    fac.synchronize()

    model.mesh.generate(2)

    model.addPhysicalGroup(1, [channel_lines[0]], inflow) # mark inflow boundary with 1
    model.addPhysicalGroup(1, [channel_lines[2]], outflow) # mark outflow boundary with 2
    model.addPhysicalGroup(1, [channel_lines[1], channel_lines[3]], walls) # mark walls with 3
    model.addPhysicalGroup(1, [circle3], noslipobstacle)
    model.addPhysicalGroup(1, [circle1, circle2], obstacle) # mark obstacle with 4
    model.addPhysicalGroup(1, [fl1, fl3], interface) # mark interface with 5
    model.addPhysicalGroup(1, [fl2], flag_tip) # mark interface with 5

    model.addPhysicalGroup(2, [plane_surface], fluid) # mark fluid domain with 6
    model.addPhysicalGroup(2, [plane_surface2], solid) # mark solid domain with 7

    nodes, topologies = msh_gmsh_model(model, dim=2)
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    gmsh.finalize()
    
    return mesh, entity_functions

# --------------------------------------------------------------------

if __name__ == '__main__':

    mesh, entity_fs = create_mesh()
    # TODO:
    # Let it return {volume_phys_tag -> [boundaing surface_tags]}
    # Create 2 submeshes with translated markers
