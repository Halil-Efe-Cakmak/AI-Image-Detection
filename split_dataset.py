import os, random, shutil

source_dir = 'dataset_raw'
target_dir = 'dataset'
classes = ['fake', 'real']
split_ratio = [0.7, 0.15, 0.15]

random.seed(42)

for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(source_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    for i, img in enumerate(images):
        src = os.path.join(class_path, img)
        if i < n_train:
            dst = os.path.join(target_dir, 'train', cls, img)
        elif i < n_train + n_val:
            dst = os.path.join(target_dir, 'val', cls, img)
        else:
            dst = os.path.join(target_dir, 'test', cls, img)
        shutil.copyfile(src, dst)

print("✅ Veri başarıyla bölündü.")
