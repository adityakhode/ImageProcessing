import cv2
import pytesseract

def detect_number(img):
    h, w = img.shape[:2]

    # ROI percentages
    x1 = int(0.088 * w)   # left
    x2 = int(0.16  * w)   # right
    y1 = int(0.83  * h)   # top
    y2 = int(0.96  * h)   # bottom
    img = img[y1:y2, x1:x2]
        # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binarize
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # slight dilation (important)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.dilate(bin_img, kernel, iterations=1)

    # resize to separate digits
    bin_img = cv2.resize(
        bin_img,
        None,
        fx=2.0,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    # OCR
    config = "--psm 6 -c tessedit_char_whitelist=0०१२३४५६७८९"

    raw = pytesseract.image_to_string(
        bin_img,
        lang="hin",
        config=config
    )
    # age = raw.split("\n")[2]
    age = raw.split("\n")[0]
    print(age)
    hindi_to_english_numbers = {
        "२१": 21, "२२": 22, "२३": 23, "२४": 24, "२५": 25,
        "२६": 26, "२७": 27, "२८": 28, "२९": 29, "३०": 30,
        "३१": 31, "३२": 32, "३३": 33, "३४": 34, "३५": 35,
        "३६": 36, "३७": 37, "३८": 38, "३९": 39, "४०": 40,
        "४१": 41, "४२": 42, "४३": 43, "४४": 44, "४५": 45,
        "४६": 46, "४७": 47, "४८": 48, "४९": 49, "५०": 50,
        "५१": 51, "५२": 52, "५३": 53, "५४": 54, "५५": 55,
        "५६": 56, "५७": 57, "५८": 58, "५९": 59, "६०": 60,
        "६१": 61, "६२": 62, "६३": 63, "६४": 64, "६५": 65,
        "६६": 66, "६७": 67, "६८": 68, "६९": 69, "७०": 70,
        "७१": 71, "७२": 72, "७३": 73, "७४": 74, "७५": 75,
        "७६": 76, "७७": 77, "७८": 78, "७९": 79, "८०": 80,
        "८१": 81, "८२": 82, "८३": 83, "८४": 84, "८५": 85,
        "८६": 86, "८७": 87, "८८": 88, "८९": 89, "९०": 90,
        "९१": 91, "९२": 92, "९३": 93, "९४": 94, "९५": 95,
        "९६": 96, "९७": 97, "९८": 98, "९९": 99, "१००": 100
    }

    return str(hindi_to_english_numbers.get(age)), bin_img