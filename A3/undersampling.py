import random
from pathlib import Path


def undersample_data():
    # Update these for your environment
    base_dir = Path('/Users/moonchaeyeon/PycharmProjects/ml_3/dataset_classification_under/train')

    waste_dir = base_dir / 'waste'
    no_waste_dir = base_dir / 'no_waste'

    waste_images = [p for p in waste_dir.glob('*.*') if p.suffix.lower() == '.jpg']
    no_waste_images = [p for p in no_waste_dir.glob('*.*') if p.suffix.lower() == '.jpg']

    target_count = len(waste_images)
    current_count = len(no_waste_images)

    print(f"Existing images")
    print(f"> Total waste images: {target_count}")
    print(f"> Total no_waste images: {current_count}")

    remove_count = current_count - target_count
    images_to_remove = random.sample(no_waste_images, remove_count)

    print(f"\nremoving {remove_count} no_waste images...")

    for img_path in images_to_remove:
        img_path.unlink()

    final_no_waste_count = len(list(no_waste_dir.glob('*.*')))
    print(f"\ndone!")
    print(f"> total waste images: {target_count}")
    print(f"> total no_waste images: {final_no_waste_count}")


if __name__ == '__main__':
    undersample_data()
