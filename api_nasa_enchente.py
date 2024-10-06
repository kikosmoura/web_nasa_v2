from flask import Flask, request, jsonify
from flask_cors import CORS
import rasterio
from pyproj import Transformer
import numpy as np

app = Flask(__name__)
CORS(app)

class RasterData:
    def __init__(self):
        self.mapbiomas_dict = self.get_dict_mapbiomas()
        self.raster_paths = {
            "enchente": "/home/s299259068/Área de Trabalho/Projetos/nasa_challenge_2024/web_nasa/geoserver/ModelAverage_Enchente_RS_v4.tif",
            "uso_solo": "/home/s299259068/Área de Trabalho/Projetos/nasa_challenge_2024/web_nasa/geoserver/Mapbiomas_RS.tif",
            "precipitacao": "/home/s299259068/Área de Trabalho/Projetos/nasa_challenge_2024/web_nasa/geoserver/preciptacao.tif",
            "elevacao": "/home/s299259068/Área de Trabalho/Projetos/nasa_challenge_2024/web_nasa/geoserver/elevacao_cropped.tif"
        }
        # Open all raster files once
        self.rasters = {}
        self.transformers = {}
        for name, path in self.raster_paths.items():
            try:
                src = rasterio.open(path)
                self.rasters[name] = src
                raster_crs = src.crs
                if raster_crs and raster_crs.is_valid:
                    self.transformers[name] = Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True)
                else:
                    self.transformers[name] = None
                    print(f"Invalid CRS for raster {name}: {raster_crs}")
            except Exception as e:
                print(f"Error opening raster {name} at {path}: {e}")

    def __del__(self):
        # Close all raster files when destroying the object
        for src in self.rasters.values():
            src.close()

    def get_dict_mapbiomas(self):
        mapbiomas_dict = {
            1: ['Floresta', 'Forest', '#32a65e'],
            3: ['Formação Florestal', 'Forest Formation', '#1f8d49'],
            4: ['Formação Savânica', 'Savanna Formation', '#7dc975'],
            5: ['Mangue', 'Mangrove', '#04381d'],
            6: ['Floresta Alagável', 'Floodable Forest', '#026975'],
            49: ['Restinga Arbórea', 'Wooded Sandbank Vegetation', '#02d659'],
            10: ['Formação Natural não Florestal', 'Non Forest Natural Formation', '#ad975a'],
            11: ['Campo Alagado e Área Pantanosa', 'Wetland', '#519799'],
            12: ['Formação Campestre', 'Grassland', '#d6bc74'],
            32: ['Apicum', 'Hypersaline Tidal Flat', '#fc8114'],
            29: ['Afloramento Rochoso', 'Rocky Outcrop', '#ffaa5f'],
            50: ['Restinga Herbácea', 'Herbaceous Sandbank Vegetation', '#ad5100'],
            13: ['Outras Formações não Florestais', 'Other non Forest Formations', '#d89f5c'],
            14: ['Agropecuária', 'Farming', '#FFFFB2'],
            15: ['Pastagem', 'Pasture', '#edde8e'],
            18: ['Agricultura', 'Agriculture', '#E974ED'],
            19: ['Lavoura Temporária', 'Temporary Crop', '#C27BA0'],
            39: ['Soja', 'Soybean', '#f5b3c8'],
            20: ['Cana', 'Sugar cane', '#db7093'],
            40: ['Arroz', 'Rice', '#c71585'],
            62: ['Algodão', 'Cotton', '#ff69b4'],
            41: ['Outras Lavouras Temporárias', 'Other Temporary Crops', '#f54ca9'],
            36: ['Lavoura Perene', 'Perennial Crop', '#d082de'],
            46: ['Café', 'Coffee', '#d68fe2'],
            47: ['Citrus', 'Citrus', '#9932cc'],
            35: ['Dendê', 'Palm Oil', '#9065d0'],
            48: ['Outras Lavouras Perenes', 'Other Perennial Crops', '#e6ccff'],
            9: ['Silvicultura', 'Forest Plantation', '#7a5900'],
            21: ['Mosaico de Usos', 'Mosaic of Uses', '#ffefc3'],
            22: ['Área não Vegetada', 'Non vegetated area', '#d4271e'],
            23: ['Praia, Duna e Areal', 'Beach, Dune and Sand Spot', '#ffa07a'],
            24: ['Área Urbanizada', 'Urban Area', '#d4271e'],
            30: ['Mineração', 'Mining', '#9c0027'],
            25: ['Outras Áreas não Vegetadas', 'Other non Vegetated Areas', '#db4d4f'],
            26: ['Corpo D\'água', 'Water', '#0000FF'],
            33: ['Rio, Lago e Oceano', 'River, Lake and Ocean', '#2532e4'],
            31: ['Aquicultura', 'Aquaculture', '#091077'],
            27: ['Não observado', 'Not Observed', '#ffffff']
        }
        return mapbiomas_dict

    def get_pixel_values_from_latlon(self, lat, lon):
        values = {}
        for name, src in self.rasters.items():
            raster_crs = src.crs
            transformer = self.transformers.get(name)
            if raster_crs is None or transformer is None:
                values[name] = 'CRS undefined or invalid in raster'
                continue
            try:
                # Transform coordinates from EPSG:4326 to raster CRS
                coords_x, coords_y = transformer.transform(lon, lat)
                # Convert to row, col indices
                row, col = src.index(coords_x, coords_y)
                # Check if indices are within raster bounds
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Read the pixel value
                    val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    if name == 'uso_solo':
                        # Get description from mapbiomas_dict
                        description = self.mapbiomas_dict.get(val, ['Desconhecido'])[1]
                        values[name] = description
                    else:
                        values[name] = float(val)
                else:
                    values[name] = None  # Coordinates out of bounds
            except Exception as e:
                values[name] = f'Error processing raster {name}: {e}'
        return values

# Create an instance of RasterData at the app level
raster_data = RasterData()

@app.route('/get_pixel_value', methods=['GET'])
def get_pixel_value():
    try:
        # Get lat and lon parameters from request
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))

        # Get pixel values from RasterData
        pixel_values = raster_data.get_pixel_values_from_latlon(lat, lon)

        # Return the values as JSON
        response = {'latitude': lat, 'longitude': lon, 'pixel_values': pixel_values}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
