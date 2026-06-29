| Date and Time | Prompt | Tool | Response Synopsis | Change |
| ------------- | ------ | ---- | ----------------- | ------ |
| 06/14/2026 5:55PM | What is the best way to calculate descriptive statistics of an image in python? | Gemini | Use OpenCV to split image into arrays, use Numpy and Scipy for stats | Imported libraries, split image into BGR channels, and calculated desc stats |
06/14/2026 8:15PM | What are ways in which I can convert an image to greyscale, binary, and (HSV, CIELAB and HLS) | Gemini | Showed function for color conversion in Opencv | Changed image to different color spaces|
| 06/14/2026 8:30PM | What is the best way to turn an image binary, where is the threshold? | Gemini | Learned about Otsu's Binarization | Implemented Otsu's Binarization |
| 06/14/2026 8:40PM | How do I apply histogram equalization to the V channel in HSV and convert it to BGR | Gemini | Explained the V channel equalization, the fix to lighting without altering hue and saturation | Normalized HSV image on V |
| 06/14/2026 8:55PM | What are random affine transformations | Gemini | Explained random affine transformations, and how to implement them randomly | Performed random affine transformations |
| 06/14/2026 9:00PM | What function do I use to apply gaussian blur using sigma values? | Gemini | pass (0,0) as kernel size | created images using sigma and gausian blur |
| 06/14/2026 9:15PM | Why do I need to greyscale images before edge detection? | Gemini | Explained edge detection math | added greyscale check before running edge detection |
| 06/14/2026 9:30PM | Explain edge detection in detail and how each method is used to test the subset? | Gemini | Broke down each of the listed methods and gave a brief explenation | Ran edge detection on the random subset
| 06/14/2026 10:00PM | How do I add 6 random plots to the readme | Gemini | Plot explenation and readme synopsis | created readme with 6 random plots
| 06/14/2026 10:15PM | How should I add processing techniques to plots | Gemini | Explained how to add processing data | Reworked script to add processing data |
| 06/28/2026 03:30PM | What is the best software on mac to create a ground truth image | Gemini | Listed possible tools | Used suggested website to create ground truth png |