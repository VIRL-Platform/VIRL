polyline_template = """
<!DOCTYPE html>
<html>
<head>
  <title>Decoded Polyline Points on Map</title>
  <style>
    /* Set the size of the div element that contains the map */
    #map {{
      height: 600px;  /* The height is 400px */
      width: 100%;  /* The width is the width of the web page */
    }}
  </style>
</head>
<body>
  <h3>My Google Maps Demo</h3>
  <!-- The div element for the map -->
  <div id="map"></div>

  <script>
    var map;

    function initMap() {{
      // Initialize the map centered on the first point of your polyline
      map = new google.maps.Map(document.getElementById('map'), {{
        zoom: 8,
        center: {{lat: 40.733359, lng: -74.002587}} // Replace with approximate center of your polyline
      }});

      // Your encoded polyline string
      var encodedPolyline = "{polyline}";

      // Check if the geometry library is properly loaded
      if (!google.maps.geometry) {{
        alert("Google Maps Geometry library failed to load");
        return;
      }}

      // Decode the polyline
      var decodedPolyline = google.maps.geometry.encoding.decodePath(encodedPolyline);

      // Place a marker for each point
      decodedPolyline.forEach(function(point, index) {{
        var marker = new google.maps.Marker({{
          position: point,
          map: map,
          icon: {{
            // Change the color of the marker here
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: '#00FF00',  // Change this to the desired color
            fillOpacity: 1,
            scale: 6,  // Adjust the size as needed
            strokeColor: 'white',
            strokeWeight: 2
          }}
        }});
        
        // Update map center to the last point
        if (index === decodedPolyline.length - 1) {{
          map.setCenter(point);
        }}
      }});


      // Draw the polyline
      var polylinePath = new google.maps.Polyline({{
        path: decodedPolyline,
        geodesic: true,
        strokeColor: '#FF0000',
        strokeOpacity: 1.0,
        strokeWeight: 2
      }});

      polylinePath.setMap(map);
    }}

    function mapsLoadError() {{
      alert("Google Maps failed to load");
    }}
  </script>

  <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initMap&libraries=geometry&onerror=mapsLoadError">
  </script>

</body>
</html>
"""

polyline_with_waypoints_template = """
<!DOCTYPE html>
<html>
<head>
  <title>Decoded Polyline Points on Map</title>
  <style>
    /* Set the size of the div element that contains the map */
    #map {{
      height: 600px;  /* The height is 400px */
      width: 100%;  /* The width is the width of the web page */
    }}
  </style>
</head>
<body>
  <h3>My Google Maps Demo</h3>
  <!-- The div element for the map -->
  <div id="map"></div>

  <script>
    var map;

    function initMap() {{
      var myStyles =[
        {{
          featureType: "poi",
          elementType: "labels",
          stylers: [
            {{ visibility: "off" }}
          ]
        }}
      ];
      
      // Initialize the map centered on the first point of your polyline
      map = new google.maps.Map(document.getElementById('map'), {{
        zoom: 8,
        center: {{lat: 40.733359, lng: -74.002587}}, // Replace with approximate center of your polyline
        styles: myStyles,
      }});

      // Your encoded polyline string
      var encodedPolyline = "{polyline}";
      var encodedWaypoints = "{waypoints}";

      // Check if the geometry library is properly loaded
      if (!google.maps.geometry) {{
        alert("Google Maps Geometry library failed to load");
        return;
      }}

      // Decode the polyline
      var decodedPolyline = google.maps.geometry.encoding.decodePath(encodedPolyline);
      var decodedWaypoints = google.maps.geometry.encoding.decodePath(encodedWaypoints);

      const image = "https://i.imgur.com/xYziheP.png";
      const svgMarker = {{
        path: "M-1.547 12l6.563-6.609-1.406-1.406-5.156 5.203-2.063-2.109-1.406 1.406zM0 0q2.906 0 4.945 2.039t2.039 4.945q0 1.453-0.727 3.328t-1.758 3.516-2.039 3.070-1.711 2.273l-0.75 0.797q-0.281-0.328-0.75-0.867t-1.688-2.156-2.133-3.141-1.664-3.445-0.75-3.375q0-2.906 2.039-4.945t4.945-2.039z",
        fillColor: "red",
        fillOpacity: 1.0,
        strokeWeight: 0,
        rotation: 0,
        scale: 2,
        anchor: new google.maps.Point(0, 20),
      }};
      
      decodedWaypoints.forEach(function(point, index) {{
        var marker = new google.maps.Marker({{
          position: point,
          map: map,
          icon: image,
        }});
        
        if (index === decodedPolyline.length - 1) {{
          map.setCenter(point);
        }}
      }});


      // Draw the polyline
      var polylinePath = new google.maps.Polyline({{
        path: decodedPolyline,
        geodesic: true,
        strokeColor: '#FF0000',
        strokeOpacity: 1.0,
        strokeWeight: 2
      }});

      polylinePath.setMap(map);
    }}

    function mapsLoadError() {{
      alert("Google Maps failed to load");
    }}
  </script>

  <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initMap&libraries=geometry&onerror=mapsLoadError">
  </script>

</body>
</html>
"""


polyline_with_waypoints_startpoint_template = """
<!DOCTYPE html>
<html>
<head>
  <title>Decoded Polyline Points on Map</title>
  <style>
    /* Set the size of the div element that contains the map */
    #map {{
      height: 600px;  /* The height is 400px */
      width: 100%;  /* The width is the width of the web page */
    }}
  </style>
</head>
<body>
  <h3>My Google Maps Demo</h3>
  <!-- The div element for the map -->
  <div id="map"></div>

  <script>
    var map;

    function initMap() {{
      var myStyles =[
        {{
          featureType: "poi",
          elementType: "labels",
          stylers: [
            {{ visibility: "off" }}
          ]
        }}
      ];
      
      // Initialize the map centered on the first point of your polyline
      map = new google.maps.Map(document.getElementById('map'), {{
        zoom: 8,
        center: {{lat: 40.733359, lng: -74.002587}}, // Replace with approximate center of your polyline
        styles: myStyles,
      }});

      // Your encoded polyline string
      var encodedPolyline = "{polyline}";
      var encodedWaypoints = "{waypoints}";
      var inputString = "{lat}, {lng}";

      var latLng = inputString.split(',');
      var latitude = parseFloat(latLng[0].trim());
      var longitude = parseFloat(latLng[1].trim());
      var startPoint = new google.maps.LatLng(latitude, longitude);

      // Check if the geometry library is properly loaded
      if (!google.maps.geometry) {{
        alert("Google Maps Geometry library failed to load");
        return;
      }}

      // Decode the polyline
      var decodedPolyline = google.maps.geometry.encoding.decodePath(encodedPolyline);
      var decodedWaypoints = google.maps.geometry.encoding.decodePath(encodedWaypoints);

      const image = "https://i.imgur.com/xYziheP.png";
      const svgMarker = {{
        path: "M-1.547 12l6.563-6.609-1.406-1.406-5.156 5.203-2.063-2.109-1.406 1.406zM0 0q2.906 0 4.945 2.039t2.039 4.945q0 1.453-0.727 3.328t-1.758 3.516-2.039 3.070-1.711 2.273l-0.75 0.797q-0.281-0.328-0.75-0.867t-1.688-2.156-2.133-3.141-1.664-3.445-0.75-3.375q0-2.906 2.039-4.945t4.945-2.039z",
        fillColor: "blue",
        fillOpacity: 1.0,
        strokeWeight: 0,
        rotation: 0,
        scale: 2,
        anchor: new google.maps.Point(0, 20),
      }};

      var marker = new google.maps.Marker({{
        position: startPoint,
        map: map,
        icon: svgMarker,
      }});
      
      decodedWaypoints.forEach(function(point, index) {{
        var marker = new google.maps.Marker({{
          position: point,
          map: map,
          icon: image,
        }});

        if (index === decodedPolyline.length - 1) {{
          map.setCenter(point);
        }}
      }});


      // Draw the polyline
      var polylinePath = new google.maps.Polyline({{
        path: decodedPolyline,
        geodesic: true,
        strokeColor: '#FF0000',
        strokeOpacity: 1.0,
        strokeWeight: 2
      }});

      polylinePath.setMap(map);
    }}

    function mapsLoadError() {{
      alert("Google Maps failed to load");
    }}
  </script>

  <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initMap&libraries=geometry&onerror=mapsLoadError">
  </script>

</body>
</html>
"""


polyline_with_waypoints_and_label_template = """
<!DOCTYPE html>
<html>
<head>
  <title>Decoded Polyline Points on Map</title>
  <style>
    /* Set the size of the div element that contains the map */
    #map {{
      height: 600px;  /* The height is 400px */
      width: 100%;  /* The width is the width of the web page */
    }}
  </style>
</head>
<body>
  <h3>My Google Maps Demo</h3>
  <!-- The div element for the map -->
  <div id="map"></div>

  <script>
    var map;

    function initMap() {{
      var myStyles =[
        {{
          featureType: "poi",
          elementType: "labels",
          stylers: [
            {{ visibility: "off" }}
          ]
        }}
      ];
      
      // Initialize the map centered on the first point of your polyline
      map = new google.maps.Map(document.getElementById('map'), {{
        zoom: 8,
        center: {{lat: 40.733359, lng: -74.002587}}, // Replace with approximate center of your polyline
        styles: myStyles,
      }});

      // Your encoded polyline string
      var encodedPolyline = "{polyline}";
      var encodedWaypoints = "{waypoints}";
      var labels = "{labels}";

      // Check if the geometry library is properly loaded
      if (!google.maps.geometry) {{
        alert("Google Maps Geometry library failed to load");
        return;
      }}

      // Decode the polyline
      var decodedPolyline = google.maps.geometry.encoding.decodePath(encodedPolyline);
      var decodedWaypoints = google.maps.geometry.encoding.decodePath(encodedWaypoints);
      var labels = labels.split(',,');

      const svgMarker = {{
        path: "M-1.547 12l6.563-6.609-1.406-1.406-5.156 5.203-2.063-2.109-1.406 1.406zM0 0q2.906 0 4.945 2.039t2.039 4.945q0 1.453-0.727 3.328t-1.758 3.516-2.039 3.070-1.711 2.273l-0.75 0.797q-0.281-0.328-0.75-0.867t-1.688-2.156-2.133-3.141-1.664-3.445-0.75-3.375q0-2.906 2.039-4.945t4.945-2.039z",
        fillColor: "red",
        fillOpacity: 1.0,
        strokeWeight: 0,
        rotation: 0,
        scale: 2,
        anchor: new google.maps.Point(0, 20),
        labelOrigin: new google.maps.Point(0, 15),
      }};
      
      decodedWaypoints.forEach(function(point, index) {{
        var marker = new google.maps.Marker({{
          position: point,
          map: map,
          icon: svgMarker,
          label: labels[index],
        }});
        
        if (index === decodedPolyline.length - 1) {{
          map.setCenter(point);
        }}
      }});


      // Draw the polyline
      var polylinePath = new google.maps.Polyline({{
        path: decodedPolyline,
        geodesic: true,
        strokeColor: '#FF0000',
        strokeOpacity: 1.0,
        strokeWeight: 2
      }});

      polylinePath.setMap(map);
    }}

    function mapsLoadError() {{
      alert("Google Maps failed to load");
    }}
  </script>

  <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initMap&libraries=geometry&onerror=mapsLoadError">
  </script>

</body>
</html>
"""
