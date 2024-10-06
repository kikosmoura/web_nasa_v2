let map;
let geocoder;
let marker;
let floodPolygons = [];
let wmsLayer; // Variável para a camada WMS
let geoJsonLayer; // Variável para a camada GeoJSON
let ucRsLayer; // Variável para a camada UC_RS
let mapbiomasLayer; // Variável para a camada Mapbiomas_RS
let elevacaoLayer; // Variável para a camada elevacao_cropped
let precipitacaoLayer; // Variável para a camada de precipitação
let hospitalsMarkers = []; // Armazenar os marcadores de hospitais
let isPointMode = false; // Variável para controlar o modo de ponto
let floodDataLayer; // Variável para a camada flood_data:limite_rs

// URL do GeoServer via ngrok com HTTPS
const wmsUrl =  'https://geoservernasa2024.sa.ngrok.io/geoserver/wms'

function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: -30.0346, lng: -51.2177 },
        zoom: 9,
    });

    geocoder = new google.maps.Geocoder();
    marker = new google.maps.Marker({
        map: map,
        draggable: false,
    });

    // Função para gerar URLs WMS com debug
    function getWmsUrl(layerName, style = '') {
        return function (coord, zoom) {
            const proj = map.getProjection();
            const zfactor = Math.pow(2, zoom);
            const top = proj.fromPointToLatLng(
                new google.maps.Point((coord.x * 256) / zfactor, (coord.y * 256) / zfactor)
            );
            const bot = proj.fromPointToLatLng(
                new google.maps.Point(((coord.x + 1) * 256) / zfactor, ((coord.y + 1) * 256) / zfactor)
            );
            const bbox = `${top.lng()},${bot.lat()},${bot.lng()},${top.lat()}`;

            const url = `${wmsUrl}?service=WMS&version=1.1.0&request=GetMap&layers=flood_data:${layerName}&styles=${style}&bbox=${bbox}&width=256&height=256&srs=EPSG:4326&format=image/png&transparent=true`;
            console.log(`WMS URL for layer ${layerName}: ${url}`); // Debug
            return url;
        };
    }

    // Inicializar camadas WMS com crossOrigin
    wmsLayer = new google.maps.ImageMapType({
        getTileUrl: getWmsUrl('ModelAverage_Enchente_v2'),
        tileSize: new google.maps.Size(256, 256),
        opacity: 0.6,
        name: 'Flood Probability',
        crossOrigin: 'anonymous', // Adicionado para CORS
    });

    ucRsLayer = new google.maps.ImageMapType({
        getTileUrl: getWmsUrl('uc_rs'),
        tileSize: new google.maps.Size(256, 256),
        opacity: 0.6,
        name: 'UC_RS',
        crossOrigin: 'anonymous', // Adicionado para CORS
    });

    mapbiomasLayer = new google.maps.ImageMapType({
        getTileUrl: getWmsUrl('Mapbiomas_RS_v2'),
        tileSize: new google.maps.Size(256, 256),
        opacity: 0.6,
        name: 'Mapbiomas_RS',
        crossOrigin: 'anonymous', // Adicionado para CORS
    });

    elevacaoLayer = new google.maps.ImageMapType({
        getTileUrl: getWmsUrl('elevacao_cropped'),
        tileSize: new google.maps.Size(256, 256),
        opacity: 0.6,
        name: 'Elevation',
        crossOrigin: 'anonymous', // Adicionado para CORS
    });

    precipitacaoLayer = new google.maps.ImageMapType({
        getTileUrl: getWmsUrl('precipitacao', 'precipitacao_blue_ramp_style'),
        tileSize: new google.maps.Size(256, 256),
        opacity: 0.6,
        name: 'Precipitation',
        crossOrigin: 'anonymous', // Adicionado para CORS
    });

    // Inicializar a camada GeoJSON
    geoJsonLayer = new google.maps.Data({ map: null });

    fetch('shape_rs_enchentes_v2.geojson')
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then((data) => {
            floodPolygons = data.features.map((feature) => feature.geometry);
            geoJsonLayer.addGeoJson(data);

            geoJsonLayer.setStyle({
                fillColor: 'blue',
                strokeColor: 'blue',
                strokeWeight: 1,
                fillOpacity: parseFloat(document.getElementById('geoJsonOpacity').value),
            });
        })
        .catch((error) => {
            console.error('Erro ao carregar GeoJSON:', error);
        });

    // Inicializar a nova camada WMS: flood_data:limite_rs
    floodDataLayer = new google.maps.ImageMapType({
        getTileUrl: getWmsUrl('limite_rs', 'styles_limites_rs'), // 'limite_rs_outline_style' é o estilo definido no GeoServer
        tileSize: new google.maps.Size(256, 256),
        opacity: 1, // Opacidade fixa
        name: 'Flood Data: Limite RS',
        crossOrigin: 'anonymous', // Adicionado para CORS
    });

    // Event Listeners
    google.maps.event.addListener(marker, 'mouseover', function () {
        const position = marker.getPosition();
        showMarkerInfo(position, marker);
    });

    document.getElementById('point-button').addEventListener('click', function () {
        isPointMode = !isPointMode;

        if (isPointMode) {
            map.setOptions({ draggableCursor: 'crosshair' });
            alert('Click on the map to add a marker.');
        } else {
            map.setOptions({ draggableCursor: 'default' });
            alert('Search mode reactivated.');
        }
    });

    // Evento para o botão de Instruções
    document.getElementById('instructions-button').addEventListener('click', function () {
        const instructionsBox = document.getElementById('instructions-box');
        if (instructionsBox.style.display === 'none' || instructionsBox.style.display === '') {
            instructionsBox.style.display = 'block';
        } else {
            instructionsBox.style.display = 'none';
        }
    });

    // Ajustar a posição do menu de camadas e opacidade para não sobrepor as novas adições
    // Se necessário, ajuste os valores de 'bottom' nos botões no CSS

    map.addListener('click', function (event) {
        if (isPointMode) {
            marker.setPosition(event.latLng);
            marker.setMap(map);
            google.maps.event.trigger(marker, 'mouseover');
        }
    });
}

function showMarkerInfo(position, markerInstance) {
    const latlng = [position.lng(), position.lat()];
    const point = turf.point(latlng);
    let isInsideFloodPolygon = false;
    let nearestDistance = Infinity;

    floodPolygons.forEach(function (layer) {
        if (turf.booleanPointInPolygon(point, layer)) {
            isInsideFloodPolygon = true;
        } else {
            const line = turf.polygonToLine(layer);
            const nearest = turf.nearestPointOnLine(line, point);
            const distance = turf.distance(point, nearest, { units: 'meters' });
            if (distance < nearestDistance) {
                nearestDistance = distance;
            }
        }
    });

    const apiUrl = `https://meuapinasa2024.sa.ngrok.io/get_pixel_value?lat=${position.lat()}&lon=${position.lng()}`;
    

    fetch(apiUrl, {
        headers: {
            'ngrok-skip-browser-warning': '1'
        }
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then((data) => {
            const pixelValues = data.pixel_values;
            const elevation = pixelValues['elevacao'];
            const precipitation = pixelValues['precipitacao'];
            const landUse = pixelValues['uso_solo'];
            const floodProbabilityValue = pixelValues['enchente'];
            const floodProbability = (floodProbabilityValue * 100).toFixed(2);

            geocoder.geocode({ location: position }, (results, status) => {
                if (status === 'OK' && results[0]) {
                    const addressComponents = results[0].address_components;
                    let street = '';
                    let number = '';
                    let neighborhood = '';
                    let city = '';
                    let postalCode = '';

                    for (const component of addressComponents) {
                        if (component.types.includes('route')) street = component.long_name;
                        if (component.types.includes('street_number')) number = component.long_name;
                        if (component.types.includes('sublocality') || component.types.includes('neighborhood'))
                            neighborhood = component.long_name;
                        if (component.types.includes('locality') || component.types.includes('administrative_area_level_2'))
                            city = component.long_name;
                        if (component.types.includes('postal_code')) postalCode = component.long_name;
                    }

                    const floodStatus = isInsideFloodPolygon
                        ? 'WITHIN the flood spot'
                        : `OUTSIDE the flood spot (Distance:: ${nearestDistance.toFixed(
                              2
                          )} meters from the nearest point of the flood spot)`;

                          const contentString = `
                          <div style="font-size: 18px; line-height: 1.5;">
                              <strong>Address:</strong> ${results[0].formatted_address}<br/>
                              <strong>Street:</strong> ${street}<br/>
                              <strong>Number:</strong> ${number}<br/>
                              <strong>Neighborhood:</strong> ${neighborhood}<br/>
                              <strong>City:</strong> ${city}<br/>
                              <strong>ZIP Code:</strong> ${postalCode}<br/>
                              <strong>Latitude:</strong> ${position.lat().toFixed(6)}<br/>
                              <strong>Longitude:</strong> ${position.lng().toFixed(6)}<br/>
                              <strong>Status:</strong> ${floodStatus}<br/>
                              <strong>Flood probability:</strong> ${floodProbability}%<br/>
                              <strong>Elevation:</strong> ${elevation.toFixed(2)} meters<br/>
                              <strong>Precipitation:</strong> ${precipitation.toFixed(2)} mm<br/>
                              <strong>Land Use:</strong> ${landUse}<br/>
                          </div>
                      `;
                      

                    const infowindow = new google.maps.InfoWindow({
                        content: contentString,
                    });
                    infowindow.open(map, markerInstance);
                }
            });
        })
        .catch((error) => {
            console.error('Erro ao buscar dados da API:', error);
            alert('Houve um problema ao buscar os dados da API.');
        });
}

function toggleWMSLayer() {
    const checkbox = document.getElementById('toggleWMS');
    if (checkbox.checked) {
        if (!map.overlayMapTypes.getArray().includes(wmsLayer)) {
            map.overlayMapTypes.push(wmsLayer);
        }
    } else {
        map.overlayMapTypes.forEach((layer, index) => {
            if (layer === wmsLayer) {
                map.overlayMapTypes.removeAt(index);
            }
        });
    }
}

function toggleGeoJSONLayer() {
    const checkbox = document.getElementById('toggleGeoJSON');
    if (checkbox.checked) {
        geoJsonLayer.setMap(map);
    } else {
        geoJsonLayer.setMap(null);
    }
}

function toggleUcRsLayer() {
    const checkbox = document.getElementById('toggleUcRs');
    if (checkbox.checked) {
        if (!map.overlayMapTypes.getArray().includes(ucRsLayer)) {
            map.overlayMapTypes.push(ucRsLayer);
        }
    } else {
        map.overlayMapTypes.forEach((layer, index) => {
            if (layer === ucRsLayer) {
                map.overlayMapTypes.removeAt(index);
            }
        });
    }
}

function toggleMapbiomasLayer() {
    const checkbox = document.getElementById('toggleMapbiomas');
    if (checkbox.checked) {
        if (!map.overlayMapTypes.getArray().includes(mapbiomasLayer)) {
            map.overlayMapTypes.push(mapbiomasLayer);
        }
    } else {
        map.overlayMapTypes.forEach((layer, index) => {
            if (layer === mapbiomasLayer) {
                map.overlayMapTypes.removeAt(index);
            }
        });
    }
}

function toggleElevacaoLayer() {
    const checkbox = document.getElementById('toggleElevacao');
    if (checkbox.checked) {
        if (!map.overlayMapTypes.getArray().includes(elevacaoLayer)) {
            map.overlayMapTypes.push(elevacaoLayer);
        }
    } else {
        map.overlayMapTypes.forEach((layer, index) => {
            if (layer === elevacaoLayer) {
                map.overlayMapTypes.removeAt(index);
            }
        });
    }
}

function togglePrecipitacaoLayer() {
    const checkbox = document.getElementById('togglePrecipitacao');
    if (checkbox.checked) {
        if (!map.overlayMapTypes.getArray().includes(precipitacaoLayer)) {
            map.overlayMapTypes.push(precipitacaoLayer);
        }
    } else {
        map.overlayMapTypes.forEach((layer, index) => {
            if (layer === precipitacaoLayer) {
                map.overlayMapTypes.removeAt(index);
            }
        });
    }
}

function toggleFloodDataLayer() {
    const checkbox = document.getElementById('toggleFloodData');
    if (checkbox.checked) {
        if (!map.overlayMapTypes.getArray().includes(floodDataLayer)) {
            map.overlayMapTypes.push(floodDataLayer);
        }
    } else {
        map.overlayMapTypes.forEach((layer, index) => {
            if (layer === floodDataLayer) {
                map.overlayMapTypes.removeAt(index);
            }
        });
    }
}

function toggleHospitalsLayer() {
    const checkbox = document.getElementById('toggleHospitals');
    if (checkbox.checked) {
        fetch('hospitals.geojson')
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                data.features.forEach((feature) => {
                    const coords = feature.geometry.coordinates;
                    const latLng = new google.maps.LatLng(coords[1], coords[0]);

                    const hospitalMarker = new google.maps.Marker({
                        position: latLng,
                        map: map,
                        icon: {
                            url: 'hospital.png',
                            scaledSize: new google.maps.Size(65, 65),
                        },
                        title: feature.properties.name || 'Hospital', // Adicionado título para melhor acessibilidade
                        // Note: Google Maps Markers não suportam a propriedade crossOrigin diretamente
                    });

                    google.maps.event.addListener(hospitalMarker, 'mouseover', function () {
                        showMarkerInfo(hospitalMarker.getPosition(), hospitalMarker);
                    });

                    hospitalsMarkers.push(hospitalMarker);
                });
            })
            .catch((error) => {
                console.error('Erro ao carregar hospitals.geojson:', error);
            });
    } else {
        hospitalsMarkers.forEach((marker) => marker.setMap(null));
        hospitalsMarkers = [];
    }
}

function changeWMSOpacity() {
    const opacity = parseFloat(document.getElementById('wmsOpacity').value);
    wmsLayer.setOpacity(opacity);
}

function changeGeoJSONOpacity() {
    const opacity = parseFloat(document.getElementById('geoJsonOpacity').value);
    geoJsonLayer.setStyle({
        fillColor: 'blue',
        strokeColor: 'blue',
        strokeWeight: 1,
        fillOpacity: opacity,
    });
}

function changeUcRsOpacity() {
    const opacity = parseFloat(document.getElementById('ucRsOpacity').value);
    ucRsLayer.setOpacity(opacity);
}

function changeMapbiomasOpacity() {
    const opacity = parseFloat(document.getElementById('mapbiomasOpacity').value);
    mapbiomasLayer.setOpacity(opacity);
}

function changeElevacaoOpacity() {
    const opacity = parseFloat(document.getElementById('elevacaoOpacity').value);
    elevacaoLayer.setOpacity(opacity);
}

function changePrecipitacaoOpacity() {
    const opacity = parseFloat(document.getElementById('precipitacaoOpacity').value);
    precipitacaoLayer.setOpacity(opacity);
}

function geocodeAddress() {
    const address = document.getElementById('address').value;
    geocoder.geocode({ address: address }, (results, status) => {
        if (status === 'OK') {
            const location = results[0].geometry.location;
            map.setCenter(location);
            map.setZoom(15);
            marker.setPosition(location);
            google.maps.event.trigger(marker, 'mouseover');
        } else {
            alert('Geocode was not successful for the following reason: ' + status);
        }
    });
}

document.getElementById('address').addEventListener('keyup', function (event) {
    if (event.key === 'Enter') {
        geocodeAddress();
    }
});

document.getElementById('search').addEventListener('click', function () {
    geocodeAddress();
});

// Evento para o botão de camadas
document.getElementById('layer-button').addEventListener('click', function () {
    const layerMenu = document.getElementById('layer-menu');
    if (layerMenu.style.display === 'none' || layerMenu.style.display === '') {
        layerMenu.style.display = 'block';
    } else {
        layerMenu.style.display = 'none';
    }
});

// Evento para o botão de opacidade
document.getElementById('opacity-button').addEventListener('click', function (event) {
    event.stopPropagation(); // Impede que o clique propague para o evento de janela
    const opacityControls = document.getElementById('opacity-controls');
    if (opacityControls.style.display === 'none' || opacityControls.style.display === '') {
        opacityControls.style.display = 'block';
    } else {
        opacityControls.style.display = 'none';
    }
});

// Evento para fechar os menus ao clicar fora
window.addEventListener('click', function (event) {
    const layerMenu = document.getElementById('layer-menu');
    const opacityControls = document.getElementById('opacity-controls');
    const instructionsBox = document.getElementById('instructions-box');
    const layerButton = document.getElementById('layer-button');
    const opacityButton = document.getElementById('opacity-button');
    const instructionsButton = document.getElementById('instructions-button');

    // Verifica se o clique não foi no layer-button ou dentro do layer-menu
    if (
        !layerMenu.contains(event.target) &&
        event.target.id !== 'layer-button' &&
        (event.target.parentElement ? event.target.parentElement.id !== 'layer-button' : true)
    ) {
        layerMenu.style.display = 'none';
    }

    // Verifica se o clique não foi no opacity-button ou dentro do opacity-controls
    if (
        !opacityControls.contains(event.target) &&
        event.target.id !== 'opacity-button' &&
        (event.target.parentElement ? event.target.parentElement.id !== 'opacity-button' : true)
    ) {
        opacityControls.style.display = 'none';
    }

    // Verifica se o clique não foi no instructions-button ou dentro do instructions-box
    if (
        !instructionsBox.contains(event.target) &&
        event.target.id !== 'instructions-button' &&
        (event.target.parentElement ? event.target.parentElement.id !== 'instructions-button' : true)
    ) {
        instructionsBox.style.display = 'none';
    }
});
window.onload = initMap;
