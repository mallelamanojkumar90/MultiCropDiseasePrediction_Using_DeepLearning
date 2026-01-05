import h5py
import os

models_dir = 'models'
for f in os.listdir(models_dir):
    if f.endswith('.h5'):
        print(f"\n--- Inspecting {f} ---")
        try:
            with h5py.File(os.path.join(models_dir, f), 'r') as h5:
                if 'layer_names' in h5.attrs:
                    print("Layer names in attributes:", [n.decode('utf-8') if isinstance(n, bytes) else n for n in h5.attrs['layer_names']])
                
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"Group: {name}")
                    elif isinstance(obj, h5py.Dataset):
                        print(f"Dataset: {name}, Shape: {obj.shape}")

                # If it's a weights file, it usually has top-level groups for layers
                print("Top level keys:", list(h5.keys()))
                # h5.visititems(print_structure)
        except Exception as e:
            print(f"Error reading {f}: {e}")
