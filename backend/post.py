import requests
import os


def upload_image(url, image_path):
    """Upload an image to the specified URL."""
    # Open the image file in binary mode
    with open(image_path, 'rb') as image_file:
        files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
        response = requests.post(url, files=files)
        return response


if __name__ == "__main__":
    # API URL
    api_url = 'http://localhost:5000/upload'  # Change to your Flask API endpoint

    # Image path
    image_path = '../images/00000013_006.png'  # Change to the actual path of the image you want to upload

    # Ensure the image file exists
    if not os.path.exists(image_path):
        print("The specified image file does not exist.")
    else:
        # Send the image to the backend
        response = upload_image(api_url, image_path)

        # Print the response
        if response.status_code == 200:
            print("Image successfully uploaded.")
            print("Server response:", response.text)
        else:
            print("Upload failed, status code:", response.status_code)
            print("Server response:", response.text)

    image_path = '../images/00000010_000.png'  # Change to the actual path of the image you want to upload

    # Ensure the image file exists
    if not os.path.exists(image_path):
        print("The specified image file does not exist.")
    else:
        # Send the image to the backend
        response = upload_image(api_url, image_path)

        # Print the response
        if response.status_code == 200:
            print("Image successfully uploaded.")
            print("Server response:", response.text)
        else:
            print("Upload failed, status code:", response.status_code)
            print("Server response:", response.text)
