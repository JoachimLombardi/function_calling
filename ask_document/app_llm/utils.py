from django.core.files.storage import default_storage



def save_file(save_path, image_file):
    if default_storage.exists(save_path):
        default_storage.delete(save_path)
    default_storage.save(save_path, image_file)
