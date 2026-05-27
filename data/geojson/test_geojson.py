import json

try:
    with open('data/geojson/ranger_mine_zones.geojson', 'r') as f:
        data = json.load(f)
    print("✓ GeoJSON file is VALID!")
    print(f"  Number of zones: {len(data['features'])}")
    
    for feature in data['features']:
        print(f"  - {feature['properties']['zone_id']}: {feature['properties']['class']}")
        
except Exception as e:
    print(f"✗ Error: {e}")
