- This program is written in Python 3.10.10

## Initialization

- Create virtual machine (VM)
  > python3 -m venv venv (also try replacing python3 with python or python3.10)
- Start VM
  > source venv/bin/activate (mac/linux)
  > venv\Scripts\activate.bat (windows)
- Install Dependencies
  > pip install -r requirements.txt
- You might get a "notice" saying you should update pip. If so, update pip then install dependencies again:
  > pip install --upgrade pip
- To initialize folder structure, simply run any command adhering to the Command Format (see "Recurring Start" section)
  > python main.py (most basic version)
- Below demonstrates the output folder structure. Note that you must have all the .py files for the program to run normally.
<img width="761" alt="Screen Shot 2023-07-02 at 12 26 05 AM" src="https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/25ec26f5-8e93-484a-a2f2-66e40866e878">

## Recurring Start

- Every new session, restart the virtual environment by running
  > source venv/bin/activate (mac/linux)
  > venv\Scripts\activate.bat (windows)
- This can be deactivated with
  > deactivate
- Put raw images in the "/img" folder. If one doesn't exist, make one or run
  > python main.py
- Command Format:
  > python main.py -n 1 -s 1 [keyword] [keyword]...
- "-n" and "-s" are flags representing the "number" of images to process and the "starting location" from which to process the images. The above example will process 1 image starting at the 2nd image file (first location is 0; the images are ordered lexigraphically, not by date! They sort similarly but are not exactly identical).
- The keywords determine what processes to run. If none is specified, all processes except "analyze" is ran.
- Keywords:
  - match, manual, fitproj, format, analyze

## Introducing the Processes

- Match: performs "template matching" which finds matches of the templates in the template folder onto the images to automate data collection
![template matching](https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/b9e03cb5-e193-47e7-8ca4-fe55b2c19ae0)

- Manual: manual intervention to edit the collected data
<img width="1104" alt="Screen Shot 2023-07-02 at 12 20 08 AM" src="https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/d30ed46a-90a7-406d-beff-e87cfeca6b5e">
![manual](https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/b4357143-6d9e-47c1-9a7e-9ca6d799219a)

- FitProj: processes collected data, identify central tooth, and converts collected data into processed data
![fit](https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/308611a5-38d0-40ae-a15e-d15b5a6eb9a0)
![projection](https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/16079a7d-0294-41fa-b216-509721e04e21)
![manual 1D](https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/ced0e3be-324a-4211-b3fe-d64fcbd4dc40)

- Format: concatenates processed data across different days and visualizes results
![eruption plot](https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/8efaafe8-9d4c-405e-9fc2-c5b3168a6ce4)

- Analyze: an interactive way for data exploration
<img width="1505" alt="Screen Shot 2023-07-02 at 12 24 33 AM" src="https://github.com/YouTelllMe/Wax-Print-Image-Processor/assets/80024712/7a2f4d64-cf9b-4f22-b6db-9b875747017a">

## Using the "Manual" GUI

1. left click: plot / unplot data
2. tab: alternates between tooth mode, gap mode, center tooth mode, center gap mode, changing the behavior of left click
3. 1, 2, 3, 4: also alternates between tooth mode, gap mode, center tooth mode, center gap mode
4. space: enter / exit viewing mode where boxes are hidden to better identify teeth
5. esc: exit without saving
6. s: exit and save
7. left arrow key: backtrack to previous image (lexigraphically) by first saving and closing the current image's GUI window
8. right arrow key: open the next image (lexigraphically) by first saving and closing the current image's GUI window

## Using the "Analyze" GUI

1. left click: open projected image of the data of the day of interest
2. left click + drag: open projected images of the data of the days of interest (stacked)
3. right click: open the manual GUI to edit data (changes not applied on plots until "format" or "analyze" is ran again)

## Additional Notices:

- Analyze first runs format then opens the interactive tool; thus running format before analyze is redundant
- Control + C and close any outstanding GUI or matplotlib windows to force exit / restart the program
- Many variable settings can be found in utils.py
