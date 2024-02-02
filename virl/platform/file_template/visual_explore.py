panorama_street_view_template = """
<!DOCTYPE html>
<html>
    <head>
    <title>Street View split-map-panes</title>

    <script src="https://polyfill.io/v3/polyfill.min.js?features=default"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <script src="https://cdn.socket.io/4.2.0/socket.io.min.js"></script>
    <script>
      var map, panorama;
      function initialize() {{
        const fenway = {{ lat: {lat}, lng: {lng} }};
        map = new google.maps.Map(document.getElementById("map"), {{
          center: fenway,
          zoom: 14,
        }});

        function render_street_view_url(position, fov, heading) {{
          var streetViewUrl = "https://maps.googleapis.com/maps/api/streetview";
          // var position = panorama.getPosition();

          var parameters = {{
            size: "300x300", // size of the image in pixels
            location: "" + position.lat + "," + position.lng, // latitude and longitude
            heading: "" + heading, // heading in degrees. 0 is north, 90 is east, etc.
            pitch: "" + 0, // specifies the up or down angle of the camera relative to the Street View vehicle
            fov: "" + fov,
            source: "outdoor",
            key: "{key}" // replace with your API key
          }};

          var url = new URL(streetViewUrl);
          Object.keys(parameters).forEach(key => url.searchParams.append(key, parameters[key]));
          // console.log(url);
          return url;
        }}

        function getStreetViewImage(position) {{
          url1 = render_street_view_url(position, 60, 0);
          url2 = render_street_view_url(position, 60, 60);
          url3 = render_street_view_url(position, 60, 120);
          url4 = render_street_view_url(position, 60, 180);
          url5 = render_street_view_url(position, 60, 240);
          url6 = render_street_view_url(position, 60, 300);

          document.getElementById('streetImages1').src = url1;
          document.getElementById('streetImages2').src = url2;
          document.getElementById('streetImages3').src = url3;
          document.getElementById('streetImages4').src = url4;
          document.getElementById('streetImages5').src = url5;
          document.getElementById('streetImages6').src = url6;
        }}

        panorama = new google.maps.StreetViewPanorama(
          document.getElementById("pano"),
          {{
            pano: '{pano_id}',
            pov: {{
              heading: {heading},
              pitch: {pitch},
            }},
          }}
        );

        document.getElementById('panorama-pov').textContent = 'Panorama Heading: ' + {heading} + ', Pitch: ' + {pitch};


        // add listener to panorama
        google.maps.event.addListener(panorama, 'position_changed', function() {{
          // var position = panorama.getPosition();
          var position;
          getStreetViewMetaDataFromPanoID(panorama.getPano(), "{key}")
            .then(location => {{
              if (location) {{
                console.log(location.lat, location.lng);
                position = location;
                // do something with location
                console.log(position);
                document.getElementById('panorama-coordinates').textContent =
                '   Panorama Latitude: ' + position.lat + ', Longitude: ' + position.lng + ' ';
                getStreetViewImage(position);

                console.log(panorama.getPano());
              }} else {{
                console.log('No location data available');
              }}
          }});

        }});

        google.maps.event.addListener(panorama, 'pov_changed', function() {{
          var pov = panorama.getPov();
          document.getElementById('panorama-pov').textContent =
            'Panorama Heading: ' + pov.heading + ', Pitch: ' + pov.pitch;
        }});

        async function getStreetViewMetaDataFromPanoID(panoId, apiKey) {{
          let url = `https://maps.googleapis.com/maps/api/streetview/metadata?pano=${{panoId}}&key=${{apiKey}}`;

          try {{
            let response = await fetch(url);

            if (!response.ok) {{
              throw new Error(`HTTP error! status: ${{response.status}}`);
            }}

            let data = await response.json();

            if (data.status === "OK") {{
              console.log(`Latitude: ${{data.location.lat}}`);
              console.log(`Longitude: ${{data.location.lng}}`);
              return data.location;
            }} else {{
              console.log('Error: ' + data.status);
              return null;
            }}

          }} catch(e) {{
            console.log('There was a problem with the fetch operation: ' + e.message);
            return null;
          }}
        }}

        map.setStreetView(panorama);
      }}

      var socket = io.connect('http://127.0.0.1:5000');

      socket.on('connect', function() {{
        console.log('Socket connected to the server');
      }});

      socket.on('disconnect', function() {{
        console.log('Sockect disconnected from the server');
      }});

      socket.on('image', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an image event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        var imageElement = document.getElementById(data.id);
        highlightElement(imageElement);
        imageElement.src = data.image;
      }});

      socket.on('image_list', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an image list event');  // Log when an event is received
        // console.log(data);  // Log the data that was received

        for (var i = 0; i < data.image_list.length; i++) {{
          // Assign the src content
          var imageElement = document.getElementById(data.id_list[i]);
          highlightElement(imageElement);
          imageElement.src = 'data:image/jpeg;base64,' + data.image_list[i];
        }}
        // var imageElement = document.getElementById(data.id);
        // imageElement.src = data.image;
      }});

      socket.on('text', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an text event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        var textElement = document.getElementById(data.id);
        highlightElement(textElement);
        textElement.textContent = data.text;
      }});

      function highlightElement(elem) {{
        // var elem = document.getElementById(id); // Get a reference to the element
        elem.classList.add("highlight"); // Add the highlight class

        setTimeout(function() {{
            elem.classList.remove("highlight"); // Remove the highlight class after 1 second
        }}, 1000);
      }}

      function clearImage(id_prefix) {{
        var i = 1;
        while (true) {{
          var imageElement = document.getElementById(id_prefix + i);

          if (imageElement) {{
              imageElement.src = "";
              i++;
          }} else {{
              // If the element doesn't exist, break out of the loop
              break;
          }}
        }}
      }}

      function clearText(id_prefix) {{
        var i = 1;
        while (true) {{
          var textElement = document.getElementById(id_prefix + i);

          if (textElement) {{
              textElement.textContent = "";
              i++;
          }} else {{
              // If the element doesn't exist, break out of the loop
              break;
          }}
        }}
      }}

      socket.on('clear', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an clear event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        clearText('VLText');
        clearImage('detectImages');
        clearImage('Road');
        // document.getElementById('RoadText').textContent = "";
        document.getElementById('VLPrompt').textContent = "";
      }});

      window.initialize = initialize;

      window.onload = function() {{
        document.getElementById("update-button").addEventListener("click", function() {{
          const newLat = parseFloat(document.getElementById("lat").value);
          const newLng = parseFloat(document.getElementById("lng").value);

          if(isNaN(newLat) || isNaN(newLng)) {{
            alert("Please provide valid lat and lng values");
          }} else {{
            getStreetViewMetaDataFromLoc(newLat, newLng, "{key}")
            .then(pano_id => {{
              if (pano_id) {{
                panorama.setOptions({{
                  pano: pano_id
                }});
                console.log(pano_id);

              }} else {{
                console.log('No panorama ID available');
              }}
          }});
        }}}}
        )
        
        document.getElementById("head-update-button").addEventListener("click", function() {{
          const newHead = parseFloat(document.getElementById("heading").value);
          if (isNaN(newHead)) {{
            alert("Please provide valid lat and lng values");
          }} else {{
            panorama.setPov({{
              heading: newHead,
              pitch: 0.0,
            }});
          }}}}
        )
      }};

      async function getStreetViewMetaDataFromLoc(lat, lng, apiKey) {{
        let url = `https://maps.googleapis.com/maps/api/streetview/metadata?location=${{lat}},${{lng}}&key=${{apiKey}}`;

        try {{
          let response = await fetch(url);

          if (!response.ok) {{
            throw new Error(`HTTP error! status: ${{response.status}}`);
          }}

          let data = await response.json();

          if (data.status === "OK") {{
            console.log(`Pano ID: ${{data.pano_id}}`);
            return data.pano_id;
          }} else {{
            console.log('Error: ' + data.status);
            return null;
          }}

        }} catch(e) {{
          console.log('There was a problem with the fetch operation: ' + e.message);
          return null;
        }}
      }}

    </script>

    <style>
    body, html {{
    margin: 10;
    padding: 0;
    height: 100%;
    }}

    #left-region {{
    float: left;
    width: 49%;
    height: 100%;
    // margin-right: 10px;
    /* display: flex;
    flex-direction: column; */
    }}

    #right-region {{
    float: right;
    width: 49%;
    height: 100%;
    display: flex;
    flex-direction: column;
    }}

    .vertical-block {{
    width: 100%;
    height: 45%;
    display: flex;
    align-items: center;
    justify-content: center;
    }}

    .vertical-block img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: cover;
    }}

    .text-block {{
    height: 10%;
    display: flex;
    align-items: center;
    justify-content: center;
    }}

    .text-block p {{
    margin: 0;
    font-weight: bold;
    text-align: center;
    }}

    #street-images,
    #detect-images {{
    overflow: hidden;
    white-space: nowrap;
    width: 100%;
    }}

    #street-images img,
    #detect-images img {{
    display: inline-block;
    max-width: 100%;
    height: auto;
    vertical-align: middle;
    }}

    .text-row {{
    flex: 0 0 auto;
    width: 100%;
    text-align: center;
    font-weight: bold;
    padding: 5px;
    }}

    .multi-column-text {{
    width: 100%;
    height: 10%;
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    }}

    .multi-column-text p {{
    width: 25%;
    }}

    .multi-column-images {{
    width: 100%;
    height: 20%;
    display: flex;
    justify-content: space-between;
    }}

    .multi-column-images img {{
    width: 25%;
    margin-bottom: 10px;
    }}

    .multi-column-text,
    .multi-column-images {{
        flex: 1 1 auto;
        display: flex;
    }}

    .multi-column-text .column,
    .multi-column-images .column {{
        width: 50%;
    }}

    .slider {{
        overflow-x: auto;
        white-space: nowrap;
    }}

    .slider img {{
        display: inline-block;
        max-width: 100%;
        height: auto;
        vertical-align: middle;
        padding: 5px;
    }}

    .slider p {{
        display: inline-block;
        max-width: 100%;
        height: auto;
        vertical-align: top;
        white-space: pre-wrap;
        padding: 5px;
    }}

    .vl-text {{
      padding-right: 5px;
    }}

    .highlight {{
        transition: background-color 1s ease;
        background-color: rgb(0, 162, 255); /* Or any color you want for the highlight */
    }}

    </style>
</head>
<body>
    <div id="left-region">
            <div class="vertical-block" id="map"></div>
            <div class="vertical-block" id="pano"></div>
            <div class="vertical-block text-block" id="information">
                <p id="panorama-coordinates"></p>
                <p id="panorama-pov"></p>
            </div>
            <!-- <div class="vertical-block" id="update-location">
              <label for="lat">Lat:</label>
              <input type="text" id="lat">
              <label for="lng">Lng:</label>
              <input type="text" id="lng">
              <button id="update-button">go</button>
            </div> -->
    </div>

    <div id="right-region">
        <div class="text-row">Street-view Images</div>
        <div id="street-images" class="multi-column-images">
            <div class="slider">
                <img id="streetImages1" alt="Street Image 1">
                <img id="streetImages2" alt="Street Image 2">
                <img id="streetImages3" alt="Street Image 3">
                <img id="streetImages4" alt="Street Image 4">
                <img id="streetImages5" alt="Street Image 5">
                <img id="streetImages6" alt="Street Image 6">
                <!-- Add more images here -->
            </div>
        </div>

        <div class="text-row">Detection Results</div>
        <div id="detect-images" class="multi-column-images">
            <div class="slider">
                <img id="detectImages1" alt="Detect Image 1">
                <img id="detectImages2" alt="Detect Image 2">
                <img id="detectImages3" alt="Detect Image 3">
                <img id="detectImages4" alt="Detect Image 4">
                <img id="detectImages5" alt="Detect Image 5">
                <img id="detectImages6" alt="Detect Image 6">
                <!-- Add more images here -->
            </div>
        </div>
        <div class="text-row">
            Vision-LLM Results
            <p id="VLPrompt" style="margin: 0px;"></p>
        </div>

        <div id="vl_answer" class="multi-column-text">
            <div class="slider">
                <p id="VLText1" class="vl-text">text 1</p>
                <p id="VLText2" class="vl-text">text 2</p>
                <p id="VLText3" class="vl-text">text 3</p>
                <p id="VLText4" class="vl-text">text 4</p>
            </div>
        </div>

        <div class="text-row">Road Selection</div>
        <div class="multi-column-images">
            <div class="slider">
                <img id="Road1" alt="Empty">
                <img id="Road2" alt="Empty">
                <img id="Road3" alt="Empty">
                <img id="Road4" alt="Empty">
                <img id="Road5" alt="Empty">
                <img id="Road6" alt="Empty">
            </div>
        </div>

        <!-- <div class="text-row" style="height: 8%;"><p id="RoadText" style="margin-top: 0;"></p></div> -->
        <div class="text-row" id="update-location">
          <label for="lat">Lat:</label>
          <input type="text" id="lat" style="width:50px;">
          <label for="lng"> Lng:</label>
          <input type="text" id="lng" style="width:50px;">
          <button id="update-button">go</button>
          <label for="heading"> Heading:</label>
          <input type="text" id="heading" style="width:50px;">
          <button id="head-update-button">adjust</button>
        </div>
    </div>

    <script
      src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initialize&v=weekly"
      defer
    ></script>

    </body>
</html>
"""

panorama_no_street_view_template = """
<!DOCTYPE html>
<html>
    <head>
    <title>Street View split-map-panes</title>

    <script src="https://polyfill.io/v3/polyfill.min.js?features=default"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <script src="https://cdn.socket.io/4.2.0/socket.io.min.js"></script>
    <script>
      var map, panorama;

      function smoothTransition(panorama, targetHeading) {{
        const startPov = panorama.getPov();

        function easeInOutCubic(t) {{
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }}

        function shortestAngleDistance(start, end) {{
            const difference = end - start;
            const alternative = (difference > 0) ? difference - 360 : difference + 360;
            return Math.abs(alternative) < Math.abs(difference) ? alternative : difference;
        }}

        const headingChange = shortestAngleDistance(startPov.heading, targetHeading);
        const frames = Math.abs(headingChange); // Use heading change as frame count
        let currentFrame = 0;

        function updatePov() {{
            if (currentFrame <= frames) {{
                let progress = easeInOutCubic(currentFrame / frames);

                // Calculate intermediate POV
                let pov = {{
                    heading: startPov.heading + progress * headingChange,
                    pitch: startPov.pitch,  // Keeping the original pitch
                    zoom: startPov.zoom     // Keeping the original zoom
                }};

                // Normalizing the heading to stay within 0-360 range
                pov.heading = (pov.heading + 360) % 360;

                panorama.setPov(pov);
                currentFrame++;
                requestAnimationFrame(updatePov);
            }}
        }}

        updatePov();
      }}

      function initialize() {{
        const fenway = {{ lat: {lat}, lng: {lng} }};
        map = new google.maps.Map(document.getElementById("map"), {{
          center: fenway,
          zoom: 14,
        }});

        function render_street_view_url(position, fov, heading) {{
          var streetViewUrl = "https://maps.googleapis.com/maps/api/streetview";
          // var position = panorama.getPosition();

          var parameters = {{
            size: "300x300", // size of the image in pixels
            location: "" + position.lat + "," + position.lng, // latitude and longitude
            heading: "" + heading, // heading in degrees. 0 is north, 90 is east, etc.
            pitch: "" + 0, // specifies the up or down angle of the camera relative to the Street View vehicle
            fov: "" + fov,
            source: "outdoor",
            key: "{key}" // replace with your API key
          }};

          var url = new URL(streetViewUrl);
          Object.keys(parameters).forEach(key => url.searchParams.append(key, parameters[key]));
          // console.log(url);
          return url;
        }}

        function getStreetViewImage(position) {{
          url1 = render_street_view_url(position, 60, 0);
          url2 = render_street_view_url(position, 60, 60);
          url3 = render_street_view_url(position, 60, 120);
          url4 = render_street_view_url(position, 60, 180);
          url5 = render_street_view_url(position, 60, 240);
          url6 = render_street_view_url(position, 60, 300);

          document.getElementById('streetImages1').src = url1;
          document.getElementById('streetImages2').src = url2;
          document.getElementById('streetImages3').src = url3;
          document.getElementById('streetImages4').src = url4;
          document.getElementById('streetImages5').src = url5;
          document.getElementById('streetImages6').src = url6;
        }}

        panorama = new google.maps.StreetViewPanorama(
          document.getElementById("pano"),
          {{
            pano: '{pano_id}',
            pov: {{
              heading: {heading},
              pitch: {pitch},
            }},
          }}
        );

        document.getElementById('panorama-pov').textContent = 'Panorama Heading: ' + {heading} + ', Pitch: ' + {pitch};


        // add listener to panorama
        google.maps.event.addListener(panorama, 'position_changed', function() {{
          // var position = panorama.getPosition();
          var position;
          getStreetViewMetaDataFromPanoID(panorama.getPano(), "{key}")
            .then(location => {{
              if (location) {{
                console.log(location.lat, location.lng);
                position = location;
                // do something with location
                console.log(position);
                document.getElementById('panorama-coordinates').textContent =
                '   Panorama Latitude: ' + position.lat + ', Longitude: ' + position.lng + ' ';

                console.log(panorama.getPano());
              }} else {{
                console.log('No location data available');
              }}
          }});

        }});

        google.maps.event.addListener(panorama, 'pov_changed', function() {{
          var pov = panorama.getPov();
          document.getElementById('panorama-pov').textContent =
            'Panorama Heading: ' + pov.heading + ', Pitch: ' + pov.pitch;
        }});

        async function getStreetViewMetaDataFromPanoID(panoId, apiKey) {{
          let url = `https://maps.googleapis.com/maps/api/streetview/metadata?pano=${{panoId}}&key=${{apiKey}}`;

          try {{
            let response = await fetch(url);

            if (!response.ok) {{
              throw new Error(`HTTP error! status: ${{response.status}}`);
            }}

            let data = await response.json();

            if (data.status === "OK") {{
              console.log(`Latitude: ${{data.location.lat}}`);
              console.log(`Longitude: ${{data.location.lng}}`);
              return data.location;
            }} else {{
              console.log('Error: ' + data.status);
              return null;
            }}

          }} catch(e) {{
            console.log('There was a problem with the fetch operation: ' + e.message);
            return null;
          }}
        }}

        map.setStreetView(panorama);
      }}

      var socket = io.connect('http://127.0.0.1:5000');

      socket.on('connect', function() {{
        console.log('Socket connected to the server');
      }});

      socket.on('disconnect', function() {{
        console.log('Sockect disconnected from the server');
      }});

      socket.on('image', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an image event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        var imageElement = document.getElementById(data.id);
        highlightElement(imageElement);
        imageElement.src = data.image;
      }});

      socket.on('image_list', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an image list event');  // Log when an event is received
        // console.log(data);  // Log the data that was received

        for (var i = 0; i < data.image_list.length; i++) {{
          // Assign the src content
          var imageElement = document.getElementById(data.id_list[i]);
          highlightElement(imageElement);
          imageElement.src = 'data:image/jpeg;base64,' + data.image_list[i];
        }}
        // var imageElement = document.getElementById(data.id);
        // imageElement.src = data.image;
      }});

      socket.on('text', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an text event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        var textElement = document.getElementById(data.id);
        highlightElement(textElement);
        textElement.textContent = data.text;
      }});

      function highlightElement(elem) {{
        // var elem = document.getElementById(id); // Get a reference to the element
        elem.classList.add("highlight"); // Add the highlight class

        setTimeout(function() {{
            elem.classList.remove("highlight"); // Remove the highlight class after 1 second
        }}, 1000);
      }}

      function clearImage(id_prefix) {{
        var i = 1;
        while (true) {{
          var imageElement = document.getElementById(id_prefix + i);

          if (imageElement) {{
              imageElement.src = "";
              i++;
          }} else {{
              // If the element doesn't exist, break out of the loop
              break;
          }}
        }}
      }}

      function clearText(id_prefix) {{
        var i = 1;
        while (true) {{
          var textElement = document.getElementById(id_prefix + i);

          if (textElement) {{
              textElement.textContent = "";
              i++;
          }} else {{
              // If the element doesn't exist, break out of the loop
              break;
          }}
        }}
      }}

      socket.on('clear', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an clear event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        clearText('VLText');
        clearImage('detectImages');
        clearImage('Road');
        // document.getElementById('RoadText').textContent = "";
        document.getElementById('VLPrompt').textContent = "";
      }});

      window.initialize = initialize;

      window.onload = function() {{
        document.getElementById("update-button").addEventListener("click", function() {{
          const newLat = parseFloat(document.getElementById("lat").value);
          const newLng = parseFloat(document.getElementById("lng").value);

          if(isNaN(newLat) || isNaN(newLng)) {{
            alert("Please provide valid lat and lng values");
          }} else {{
            getStreetViewMetaDataFromLoc(newLat, newLng, "{key}")
            .then(pano_id => {{
              if (pano_id) {{
                panorama.setOptions({{
                  pano: pano_id
                }});
                console.log(pano_id);

              }} else {{
                console.log('No panorama ID available');
              }}
          }});
        }}}}
        )

        document.getElementById("head-update-button").addEventListener("click", function() {{
          const newHead = parseFloat(document.getElementById("heading").value);
          if (isNaN(newHead)) {{
            alert("Please provide valid lat and lng values");
          }} else {{
            smoothTransition(panorama, newHead);
          }}}}
        )
      }};

      async function getStreetViewMetaDataFromLoc(lat, lng, apiKey) {{
        let url = `https://maps.googleapis.com/maps/api/streetview/metadata?location=${{lat}},${{lng}}&key=${{apiKey}}`;

        try {{
          let response = await fetch(url);

          if (!response.ok) {{
            throw new Error(`HTTP error! status: ${{response.status}}`);
          }}

          let data = await response.json();

          if (data.status === "OK") {{
            console.log(`Pano ID: ${{data.pano_id}}`);
            return data.pano_id;
          }} else {{
            console.log('Error: ' + data.status);
            return null;
          }}

        }} catch(e) {{
          console.log('There was a problem with the fetch operation: ' + e.message);
          return null;
        }}
      }}

    </script>

    <style>
    body, html {{
    margin: 10;
    padding: 0;
    height: 100%;
    }}

    #left-region {{
    float: left;
    width: 49%;
    height: 100%;
    // margin-right: 10px;
    /* display: flex;
    flex-direction: column; */
    }}

    #right-region {{
    float: right;
    width: 49%;
    height: 100%;
    display: flex;
    flex-direction: column;
    }}

    .vertical-block {{
    width: 100%;
    height: 45%;
    display: flex;
    align-items: center;
    justify-content: center;
    }}

    .vertical-block img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: cover;
    }}

    .text-block {{
    height: 10%;
    display: flex;
    align-items: center;
    justify-content: center;
    }}

    .text-block p {{
    margin: 0;
    font-weight: bold;
    text-align: center;
    }}

    #street-images,
    #detect-images {{
    overflow: hidden;
    white-space: nowrap;
    width: 100%;
    }}

    #street-images img,
    #detect-images img {{
    display: inline-block;
    max-width: 100%;
    height: auto;
    vertical-align: middle;
    }}

    .text-row {{
    flex: 0 0 auto;
    width: 100%;
    text-align: center;
    font-weight: bold;
    padding: 5px;
    }}

    .multi-column-text {{
    width: 100%;
    height: 10%;
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    }}

    .multi-column-text p {{
    width: 25%;
    }}

    .multi-column-images {{
    width: 100%;
    height: 20%;
    display: flex;
    justify-content: space-between;
    }}

    .multi-column-images img {{
    width: 25%;
    margin-bottom: 10px;
    }}

    .multi-column-text,
    .multi-column-images {{
        flex: 1 1 auto;
        display: flex;
    }}

    .multi-column-text .column,
    .multi-column-images .column {{
        width: 50%;
    }}

    .slider {{
        overflow-x: auto;
        white-space: nowrap;
    }}

    .slider img {{
        display: inline-block;
        max-width: 100%;
        height: auto;
        vertical-align: middle;
        padding: 5px;
    }}

    .slider p {{
        display: inline-block;
        max-width: 100%;
        height: auto;
        vertical-align: top;
        white-space: pre-wrap;
        padding: 5px;
    }}

    .vl-text {{
      padding-right: 5px;
    }}

    .highlight {{
        transition: background-color 1s ease;
        background-color: rgb(0, 162, 255); /* Or any color you want for the highlight */
    }}

    </style>
</head>
<body>
    <div id="left-region">
            <div class="vertical-block" id="map"></div>
            <div class="vertical-block" id="pano"></div>
            <div class="vertical-block text-block" id="information">
                <p id="panorama-coordinates"></p>
                <p id="panorama-pov"></p>
            </div>
            <!-- <div class="vertical-block" id="update-location">
              <label for="lat">Lat:</label>
              <input type="text" id="lat">
              <label for="lng">Lng:</label>
              <input type="text" id="lng">
              <button id="update-button">go</button>
            </div> -->
    </div>

    <div id="right-region">
        <div class="text-row">Street-view Images</div>
        <div id="street-images" class="multi-column-images">
            <div class="slider">
                <img id="streetImages1" alt="Street Image 1">
                <img id="streetImages2" alt="Street Image 2">
                <img id="streetImages3" alt="Street Image 3">
                <img id="streetImages4" alt="Street Image 4">
                <img id="streetImages5" alt="Street Image 5">
                <img id="streetImages6" alt="Street Image 6">
                <!-- Add more images here -->
            </div>
        </div>

        <div class="text-row">Detection Results</div>
        <div id="detect-images" class="multi-column-images">
            <div class="slider">
                <img id="detectImages1" alt="Awaiting Detection Image">
                <img id="detectImages2" alt="Awaiting Detection Image">
                <img id="detectImages3" alt="Awaiting Detection Image">
                <img id="detectImages4" alt="Awaiting Detection Image">
                <img id="detectImages5" alt="Awaiting Detection Image">
                <img id="detectImages6" alt="Awaiting Detection Image">
                <!-- Add more images here -->
            </div>
        </div>
        <div class="text-row">
            Vision-LLM Results
            <p id="VLPrompt" style="margin: 0px;"></p>
        </div>

        <div id="vl_answer" class="multi-column-text">
            <div class="slider">
                <p id="VLText1" class="vl-text">text 1</p>
                <p id="VLText2" class="vl-text">text 2</p>
                <p id="VLText3" class="vl-text">text 3</p>
                <p id="VLText4" class="vl-text">text 4</p>
            </div>
        </div>

        <div class="text-row">Road Selection</div>
        <div class="multi-column-images">
            <div class="slider">
                <img id="Road1" alt="Empty">
                <img id="Road2" alt="Empty">
                <img id="Road3" alt="Empty">
                <img id="Road4" alt="Empty">
                <img id="Road5" alt="Empty">
                <img id="Road6" alt="Empty">
            </div>
        </div>

        <!-- <div class="text-row" style="height: 8%;"><p id="RoadText" style="margin-top: 0;"></p></div> -->
        <div class="text-row" id="update-location">
          <label for="lat">Lat:</label>
          <input type="text" id="lat" style="width:50px;">
          <label for="lng"> Lng:</label>
          <input type="text" id="lng" style="width:50px;">
          <button id="update-button">go</button>
          <label for="heading"> Heading:</label>
          <input type="text" id="heading" style="width:50px;">
          <button id="head-update-button">adjust</button>
        </div>
    </div>

    <script
      src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initialize&v=weekly"
      defer
    ></script>

    </body>
</html>
"""

detect_no_street_view_no_pana_control_template = """
<!DOCTYPE html>
<html>
    <head>
    <title>Street View split-map-panes</title>

    <script src="https://polyfill.io/v3/polyfill.min.js?features=default"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <script src="https://cdn.socket.io/4.2.0/socket.io.min.js"></script>
    <script>
      var map, panorama;

      function smoothTransition(panorama, targetHeading) {{
        const startPov = panorama.getPov();

        function easeInOutCubic(t) {{
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }}

        function shortestAngleDistance(start, end) {{
            const difference = end - start;
            const alternative = (difference > 0) ? difference - 360 : difference + 360;
            return Math.abs(alternative) < Math.abs(difference) ? alternative : difference;
        }}

        const headingChange = shortestAngleDistance(startPov.heading, targetHeading);
        const frames = Math.abs(headingChange); // Use heading change as frame count
        let currentFrame = 0;

        function updatePov() {{
            if (currentFrame <= frames) {{
                let progress = easeInOutCubic(currentFrame / frames);

                // Calculate intermediate POV
                let pov = {{
                    heading: startPov.heading + progress * headingChange,
                    pitch: startPov.pitch,  // Keeping the original pitch
                    zoom: startPov.zoom     // Keeping the original zoom
                }};

                // Normalizing the heading to stay within 0-360 range
                pov.heading = (pov.heading + 360) % 360;

                panorama.setPov(pov);
                currentFrame++;
                requestAnimationFrame(updatePov);
            }}
        }}

        updatePov();
      }}

      function initialize() {{
        const fenway = {{ lat: {lat}, lng: {lng} }};
        map = new google.maps.Map(document.getElementById("map"), {{
          center: fenway,
          zoom: 14,
        }});

        function render_street_view_url(position, fov, heading) {{
          var streetViewUrl = "https://maps.googleapis.com/maps/api/streetview";
          // var position = panorama.getPosition();

          var parameters = {{
            size: "300x300", // size of the image in pixels
            location: "" + position.lat + "," + position.lng, // latitude and longitude
            heading: "" + heading, // heading in degrees. 0 is north, 90 is east, etc.
            pitch: "" + 0, // specifies the up or down angle of the camera relative to the Street View vehicle
            fov: "" + fov,
            source: "outdoor",
            key: "{key}" // replace with your API key
          }};

          var url = new URL(streetViewUrl);
          Object.keys(parameters).forEach(key => url.searchParams.append(key, parameters[key]));
          // console.log(url);
          return url;
        }}

        function getStreetViewImage(position) {{
          url1 = render_street_view_url(position, 60, 0);
          url2 = render_street_view_url(position, 60, 60);
          url3 = render_street_view_url(position, 60, 120);
          url4 = render_street_view_url(position, 60, 180);
          url5 = render_street_view_url(position, 60, 240);
          url6 = render_street_view_url(position, 60, 300);

          document.getElementById('streetImages1').src = url1;
          document.getElementById('streetImages2').src = url2;
          document.getElementById('streetImages3').src = url3;
          document.getElementById('streetImages4').src = url4;
          document.getElementById('streetImages5').src = url5;
          document.getElementById('streetImages6').src = url6;
        }}

        panorama = new google.maps.StreetViewPanorama(
          document.getElementById("pano"),
          {{
            pano: '{pano_id}',
            pov: {{
              heading: {heading},
              pitch: {pitch},
            }},
          }}
        );

        panorama_v2 = new google.maps.StreetViewPanorama(
          document.getElementById("pano-vis"),
          {{
            pano: 'kkSs6xQM_4zk8ibY5gouBw',
            pov: {{
              heading: 0.0,
              pitch: 0.0,
            }},
            zoomControl: false,
            addressControl: false,
            fullscreenControl: false,
            panControl: false,
            clickToGo: false,
            linksControl: false,
            showRoadLabels: false,
          }}
        );

        document.getElementById('panorama-pov').textContent = 'Panorama Heading: ' + {heading} + ', Pitch: ' + {pitch};


        // add listener to panorama
        google.maps.event.addListener(panorama, 'position_changed', function() {{
          // var position = panorama.getPosition();
          var position;
          getStreetViewMetaDataFromPanoID(panorama.getPano(), "{key}")
            .then(location => {{
              if (location) {{
                position = location;
                // do something with location
                document.getElementById('panorama-coordinates').textContent =
                '   Panorama Latitude: ' + position.lat + ', Longitude: ' + position.lng + ' ';
                panorama_v2.setPano(panorama.getPano())
              }} else {{
                console.log('No location data available');
              }}
          }});

        }});

        google.maps.event.addListener(panorama, 'pov_changed', function() {{
          var pov = panorama.getPov();
          document.getElementById('panorama-pov').textContent =
            'Panorama Heading: ' + pov.heading + ', Pitch: ' + pov.pitch;
          panorama_v2.setPov(pov)
        }});

        async function getStreetViewMetaDataFromPanoID(panoId, apiKey) {{
          let url = `https://maps.googleapis.com/maps/api/streetview/metadata?pano=${{panoId}}&key=${{apiKey}}`;

          try {{
            let response = await fetch(url);

            if (!response.ok) {{
              throw new Error(`HTTP error! status: ${{response.status}}`);
            }}

            let data = await response.json();

            if (data.status === "OK") {{
              console.log(`Latitude: ${{data.location.lat}}`);
              console.log(`Longitude: ${{data.location.lng}}`);
              return data.location;
            }} else {{
              console.log('Error: ' + data.status);
              return null;
            }}

          }} catch(e) {{
            console.log('There was a problem with the fetch operation: ' + e.message);
            return null;
          }}
        }}

        map.setStreetView(panorama);
      }}

      var socket = io.connect('http://127.0.0.1:5000');

      socket.on('connect', function() {{
        console.log('Socket connected to the server');
      }});

      socket.on('disconnect', function() {{
        console.log('Sockect disconnected from the server');
      }});

      socket.on('image', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an image event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        var imageElement = document.getElementById(data.id);
        highlightElement(imageElement);
        imageElement.src = data.image;
      }});

      socket.on('image_list', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an image list event');  // Log when an event is received
        // console.log(data);  // Log the data that was received

        for (var i = 0; i < data.image_list.length; i++) {{
          // Assign the src content
          var imageElement = document.getElementById(data.id_list[i]);
          highlightElement(imageElement);
          imageElement.src = 'data:image/jpeg;base64,' + data.image_list[i];
        }}
        // var imageElement = document.getElementById(data.id);
        // imageElement.src = data.image;
      }});

      socket.on('text', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an text event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        var textElement = document.getElementById(data.id);
        highlightElement(textElement);
        textElement.textContent = data.text;
      }});

      function highlightElement(elem) {{
        // var elem = document.getElementById(id); // Get a reference to the element
        elem.classList.add("highlight"); // Add the highlight class

        setTimeout(function() {{
            elem.classList.remove("highlight"); // Remove the highlight class after 1 second
        }}, 1000);
      }}

      function clearImage(id_prefix) {{
        var i = 1;
        while (true) {{
          var imageElement = document.getElementById(id_prefix + i);

          if (imageElement) {{
              imageElement.src = "";
              i++;
          }} else {{
              // If the element doesn't exist, break out of the loop
              break;
          }}
        }}
      }}

      function clearText(id_prefix) {{
        var i = 1;
        while (true) {{
          var textElement = document.getElementById(id_prefix + i);

          if (textElement) {{
              textElement.textContent = "";
              i++;
          }} else {{
              // If the element doesn't exist, break out of the loop
              break;
          }}
        }}
      }}

      socket.on('clear', function(data) {{
        // socket.emit('messageReceived');
        console.log('Socket received an clear event');  // Log when an event is received
        // console.log(data);  // Log the data that was received
        clearText('VLText');
        clearImage('detectImages');
        clearImage('Road');
        // document.getElementById('RoadText').textContent = "";
        document.getElementById('VLPrompt').textContent = "";
      }});

      window.initialize = initialize;

      window.onload = function() {{
        document.getElementById("update-button").addEventListener("click", function() {{
          const newLat = parseFloat(document.getElementById("lat").value);
          const newLng = parseFloat(document.getElementById("lng").value);

          if(isNaN(newLat) || isNaN(newLng)) {{
            alert("Please provide valid lat and lng values");
          }} else {{
            getStreetViewMetaDataFromLoc(newLat, newLng, "{key}")
            .then(pano_id => {{
              if (pano_id) {{
                panorama.setOptions({{
                  pano: pano_id
                }});
                console.log(pano_id);

              }} else {{
                console.log('No panorama ID available');
              }}
          }});
        }}}}
        )

        document.getElementById("head-update-button").addEventListener("click", function() {{
          const newHead = parseFloat(document.getElementById("heading").value);
          if (isNaN(newHead)) {{
            alert("Please provide valid lat and lng values");
          }} else {{
            smoothTransition(panorama, newHead);
          }}}}
        )
      }};

      async function getStreetViewMetaDataFromLoc(lat, lng, apiKey) {{
        let url = `https://maps.googleapis.com/maps/api/streetview/metadata?location=${{lat}},${{lng}}&key=${{apiKey}}`;

        try {{
          let response = await fetch(url);

          if (!response.ok) {{
            throw new Error(`HTTP error! status: ${{response.status}}`);
          }}

          let data = await response.json();

          if (data.status === "OK") {{
            console.log(`Pano ID: ${{data.pano_id}}`);
            return data.pano_id;
          }} else {{
            console.log('Error: ' + data.status);
            return null;
          }}

        }} catch(e) {{
          console.log('There was a problem with the fetch operation: ' + e.message);
          return null;
        }}
      }}

    </script>

    <style>
    body, html {{
    margin: 10;
    padding: 0;
    height: 100%;
    }}

    #left-region {{
    float: left;
    width: 49%;
    height: 100%;
    // margin-right: 10px;
    /* display: flex;
    flex-direction: column; */
    }}

    #right-region {{
    float: right;
    width: 49%;
    height: 100%;
    display: flex;
    flex-direction: column;
    }}

    .vertical-block {{
    width: 100%;
    height: 45%;
    display: flex;
    align-items: center;
    justify-content: center;
    }}

    .vertical-block img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: cover;
    }}

    .text-block {{
    height: 10%;
    display: flex;
    align-items: center;
    justify-content: center;
    }}

    .text-block p {{
    margin: 0;
    font-weight: bold;
    text-align: center;
    }}

    #street-images,
    #detect-images {{
    overflow: hidden;
    white-space: nowrap;
    width: 100%;
    }}

    #street-images img,
    #detect-images img {{
    display: inline-block;
    max-width: 100%;
    height: auto;
    vertical-align: middle;
    }}

    .text-row {{
    flex: 0 0 auto;
    width: 100%;
    text-align: center;
    font-weight: bold;
    padding: 5px;
    }}

    .multi-column-text {{
    width: 100%;
    height: 10%;
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    }}

    .multi-column-text p {{
    width: 25%;
    }}

    .multi-column-images {{
    width: 100%;
    height: 20%;
    display: flex;
    justify-content: space-between;
    }}

    .multi-column-images img {{
    width: 25%;
    margin-bottom: 10px;
    }}

    .multi-column-text,
    .multi-column-images {{
        flex: 1 1 auto;
        display: flex;
    }}

    .multi-column-text .column,
    .multi-column-images .column {{
        width: 50%;
    }}

    .slider {{
        overflow-x: auto;
        white-space: nowrap;
    }}

    .slider img {{
        display: inline-block;
        max-width: 100%;
        height: auto;
        vertical-align: middle;
        padding: 5px;
    }}

    .slider p {{
        display: inline-block;
        max-width: 100%;
        height: auto;
        vertical-align: top;
        white-space: pre-wrap;
        padding: 5px;
    }}

    .vl-text {{
      padding-right: 5px;
    }}

    .highlight {{
        transition: background-color 1s ease;
        background-color: rgb(0, 162, 255); /* Or any color you want for the highlight */
    }}

    </style>
</head>
<body>
    <div id="left-region">
            <div class="vertical-block" id="map"></div>
            <div class="vertical-block" id="pano"></div>
            <div class="vertical-block text-block" id="information">
                <p id="panorama-coordinates"></p>
                <p id="panorama-pov"></p>
            </div>
            <!-- <div class="vertical-block" id="update-location">
              <label for="lat">Lat:</label>
              <input type="text" id="lat">
              <label for="lng">Lng:</label>
              <input type="text" id="lng">
              <button id="update-button">go</button>
            </div> -->
    </div>

    <div id="right-region">
        <div class="text-row">Detection Results</div>
        <div id="detect-images" class="multi-column-images">
            <div class="slider">
                <img id="detectImages1" alt="Awaiting Detection Image">
                <img id="detectImages2" alt="Awaiting Detection Image">
                <img id="detectImages3" alt="Awaiting Detection Image">
                <img id="detectImages4" alt="Awaiting Detection Image">
                <img id="detectImages5" alt="Awaiting Detection Image">
                <img id="detectImages6" alt="Awaiting Detection Image">
                <!-- Add more images here -->
            </div>
        </div>
        <div class="vertical-block" id="pano-vis"><p id="panorama-pov-vis"></p></div>

        <!-- <div class="text-row" style="height: 8%;"><p id="RoadText" style="margin-top: 0;"></p></div> -->
        <div class="text-row" id="update-location">
          <label for="lat">Lat:</label>
          <input type="text" id="lat" style="width:50px;">
          <label for="lng"> Lng:</label>
          <input type="text" id="lng" style="width:50px;">
          <button id="update-button">go</button>
          <label for="heading"> Heading:</label>
          <input type="text" id="heading" style="width:50px;">
          <button id="head-update-button">adjust</button>
        </div>
    </div>

    <script
      src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initialize&v=weekly"
      defer
    ></script>

    </body>
</html>
"""
