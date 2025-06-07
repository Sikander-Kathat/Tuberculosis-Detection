<?php

error_reporting(E_ALL);
ini_set('display_errors', 1);

// Check if the form was submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Check if a file was uploaded
    if (isset($_FILES["image"])) {
        $file = $_FILES["image"];

        // Check if the file is an image
        $fileType = exif_imagetype($file["tmp_name"]);
        if ($fileType !== false) {
            // Save the uploaded image to the specified folder
            $uploadFolder = "C:\\xampp\\htdocs\\Tuberculosis_detection\\UPLOADED_IMAGE\\";
            $destination = $uploadFolder . $file["name"];
            if (move_uploaded_file($file["tmp_name"], $destination)) {
                // Output HTML header
                echo <<<HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <!-- Mobile Metas -->
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <!-- Site Metas -->
    <meta name="keywords" content="" />
    <meta name="description" content="" />
    <meta name="author" content="" />

    <title>Tuberculosis Detection AI</title>

    <!-- slider stylesheet -->
    <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/OwlCarousel2/2.1.3/assets/owl.carousel.min.css" />

    <!-- bootstrap core css -->
    <link rel="stylesheet" type="text/css" href="css/bootstrap.css" />

    <!-- fonts style -->
    <link href="https://fonts.googleapis.com/css?family=Poppins:400,600,700&display=swap" rel="stylesheet">
    <!-- Custom styles for this template -->
    <link href="css/style.css" rel="stylesheet" />
    <!-- responsive style -->
    <link href="css/responsive.css" rel="stylesheet" />
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa; /* Keep background color same as the provided page */
        }

        h1 {
            font-size: 36px;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }

        .result {
            font-size: 24px;
            color: #007bff; /* Keep font color same as Tuberculosis Detected */
            text-align: center;
            margin-top: 20px; /* Adjust margin for spacing */
        }

        .go-back {
            text-align: center;
            margin-top: 20px;
        }

        .go-back a {
            color: #007bff;
            text-decoration: none;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <!-- header section strats -->
    <header class="header_section">
      <div class="container">
        <nav class="navbar navbar-expand-lg custom_nav-container ">
          <a class="navbar-brand" href="index.html">
            <img src="images/logo-removebg-preview.png" alt="">
            <span>
              Tuberculosis Detection
            </span>
          </a>
          <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
            <span class="s-1"> </span>
            <span class="s-2"> </span>
            <span class="s-3"> </span>
          </button>

          <div class="collapse navbar-collapse" id="navbarSupportedContent">
            <div class="d-flex ml-auto flex-column flex-lg-row align-items-center">
              <ul class="navbar-nav  ">
                <li class="nav-item active">
                  <a class="nav-link" href="index.html"> Home <span class="sr-only">(current)</span></a>
                </li>
                <li class="nav-item">
                  <a class="nav-link" href="about.html"> About</a>
                </li>
                <li class="nav-item">
                  <a class="nav-link" href="service.html"> Service </a>
                </li>
                <li class="nav-item">
                  <a class="nav-link" href="model.html"> AI Models </a>
                </li>
                <li class="nav-item">
                  <a class="nav-link" href="contact.html">Contact </a>
                </li>
                <li class="nav-item">
                  <a class="nav-link" href="./logout.php" class="btn btn-primary">Log Out</a>
                </li>
              </ul>
            </div>
          </div>
        </nav>
      </div>
    </header>
    <div class="container">
HTML;

                // Call the Python script with the uploaded image as an argument
                $imagePath = escapeshellarg($destination); // escape special characters
                exec("python detect.py $imagePath", $output, $return_code);
                
                // Interpret return code and display appropriate message
                if ($return_code === 0) {
                    echo "<p class='result'>Tuberculosis not Detected</p>";
                } elseif ($return_code === 1) {
                    echo "<p class='result'>Tuberculosis Detected</p>";
                } else {
                    echo "<p class='result'>Error occurred during detection</p>";
                }

                // Output "Go Back" button
                echo <<<HTML
                <div class="go-back">
                  <a href="detect.html">Go Back</a>
                </div>
                </div>
            </body>
            </html>
HTML;

            } else {
                echo "Failed to move uploaded file.";
            }
        } else {
            echo "Invalid file format. Please upload an image.";
        }
    } else {
        echo "No file uploaded.";
    }
} else {
    // Redirect to the homepage if accessed directly
    header("Location: index.html");
    exit;
}

?>
