import bpy
import json
import mathutils
import os

# Load FBX file
fbx_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bim', 'dicelab_bim.fbx')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_file)

object_data = {}

for obj in bpy.context.scene.objects:
    if obj.type in ['MESH', 'EMPTY']:
        # Use bounding box center in world space for MESH
        if obj.type == 'MESH':
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            center = sum(bbox_corners, mathutils.Vector()) / 8.0
            location = list(center)
        else:
            location = list(obj.location)

        dimensions = list(obj.dimensions) if obj.type == 'MESH' else [0, 0, 0]
        rotation = list(obj.rotation_euler) if hasattr(obj, 'rotation_euler') else [0, 0, 0]

        # Prepare a normalized name or category
        family_name = obj.name.lower().split('.')[0]  # remove .001, .002 etc.

        object_data[obj.name] = {
            'name': obj.name,
            'type': obj.type,
            'location': location,
            'dimensions': dimensions,
            'rotation': rotation,
            'family_name': family_name,
            'danger_coefficient': None,         
            'bayesian_coefficient': None,      
            'class_name': None                 
        }

# Save the JSON

output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils', 'extracted_objects1.json')
with open(output_file, 'w') as f:
    json.dump(object_data, f, indent=4)

print(f"Extracted {len(object_data)} objects with enriched metadata.")
