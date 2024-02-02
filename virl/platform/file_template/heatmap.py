heatmap_template = """
<!DOCTYPE html>
<html>
  <head>
    <title>Heatmap Example</title>
    <script src="https://maps.googleapis.com/maps/api/js?key={key}&libraries=visualization"></script>
    <style>
      #map {
        height: 400px;
        width: 100%;
      }
    </style>
  </head>
  <body>
    <div id="map"></div>
    <script>
      function initMap() {{
        var map = new google.maps.Map(document.getElementById('map'), {{
          zoom: 9,
          center: {{lat: {lat}, lng: {lng} }},
        }});

        // Fetch data from Flask server endpoint
        fetch('./heatmap_data.txt')
          .then(response => response.text())
          .then(text => {{
            var lines = text.split('\n');
            var heatmapData = lines.map(line => {{
              var parts = line.replace('(', '').replace(')', '').split(', ');
              return {{location: new google.maps.LatLng(parseFloat(parts[0]), parseFloat(parts[1])), weight: 1}};
            }});

            var heatmap = new google.maps.visualization.HeatmapLayer({{
              data: heatmapData
            }});
            heatmap.setMap(map);
          }});
      }}

      google.maps.event.addDomListener(window, 'load', initMap);
    </script>
  </body>
</html>
"""
