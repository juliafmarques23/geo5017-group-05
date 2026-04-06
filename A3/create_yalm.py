import yaml

# Define the data structure
data = {
    'path': './datasets',       
    'train': 'train/images',
    'val': 'valid/images',
    'nc': 5,
    'names': [
        'bulky waste',
        'garbage bag',
        'cardboard',
        'litter',
        'other'
    ]
}

# Write the structure to the data.yaml file
with open('data.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print("data.yaml has been created successfully!")