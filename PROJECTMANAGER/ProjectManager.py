import hou
import os
from hutil.Qt import QtCore, QtWidgets, QtUiTools
import json

"""
    This is a world of pain for our gain.
    Setup up JobManager in Houdini To load jobs easier
    TO DO
        Load currentJob on launch
        Try a form Layout to see if it scales better
        Save a snapshot of the Viewport
        List current scene files in the folder
        sort 
        add an open file option

"""
light_label_style = " color: #d4b56c; font-size: 18px"
border_label_style = "border: 1px; solid black; color: #d4b59a;"

#local path to the script
scriptpath =  os.path.dirname(__file__)
class ProjectManager(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.jobs = []
        self.setMinimumSize(600, 800)
        #load UI file
        loader = QtUiTools.QUiLoader()
        self.ui = loader.load(f"{scriptpath}/UI/Projectmanager_01.ui")
        self.ui.lab_heading.setStyleSheet(light_label_style)
        self.ui.lab_currentPath.setStyleSheet(border_label_style)
        self.currentJob = None
        self.getCurrentJob()
        self.ui.but_selectBaseFolder.clicked.connect(self.handleChooseDirectories)
        self.ui.but_LoadProject.clicked.connect(self.loadSelectedJob)
        self.ui.but_SaveDescription.clicked.connect(self.saveDescription)

        # layout
        mainLayout = QtWidgets.QVBoxLayout()    #ge nerate a layout for houdini
        mainLayout.setContentsMargins(0,0,0,0)  
        mainLayout.addWidget(self.ui)   #add ui to widget
        self.setLayout(mainLayout)

    def handleChooseDirectories(self):
        baseFolder = hou.ui.selectFile(start_directory = self.currentJob, file_type= hou.fileType.Directory)
        if not os.path.exists(baseFolder):
           hou.ui.displayMessage("Select a valid path")
           return
        hou.ui.displayMessage(f"You Chose the Folder with Jobs:\n < {baseFolder} >", buttons=("ok",))
        self.ui.led_projectPath.setText(baseFolder)
        self.loadJobDirs()


    def loadJobDirs(self):
        self.ui.lw_jobList.clear()
        self.jobs = [f.name for f in os.scandir(self.ui.led_projectPath.text()) if f.is_dir()]
        self.ui.lw_jobList.addItems(self.jobs)
        # self.ui.lab_currentPath.setText(self.currentJob)

    def loadSelectedJob(self):        
        try:
            jobname = f"{self.ui.led_projectPath.text()}{self.ui.lw_jobList.currentItem().text()}"
            hou.allowEnvironmentToOverwriteVariable("JOB", True)
            hou.putenv("JOB",jobname)
        except Exception as E:
            hou.ui.displayMessage(f"Failed to set job\n {E} \n\n Try Selecting a Job", buttons=("ok",))
        else:
            self.currentJob = jobname
            self.ui.lab_currentPath.setText(jobname)
            #self.ui.led_projectPath.setSelected()
            self.loadDescription()
            hou.ui.setStatusMessage(f"Changed the job to:\n <  {jobname.split('/')[-1]}  >", severity = hou.severityType.ImportantMessage)
            
        

    def getCurrentJob(self):       
        if self.currentJob == None:
                jobPath = hou.getenv("MYJOBS")
                if jobPath == None:
                    self.jobPath = "C:/Users/tkewl/Documents/houdini20.0"
                self.ui.led_projectPath.setText(jobPath)
                self.loadJobDirs()
        else:
            self.ui.led_projectPath.setText("/".join(self.currentJob.split("/")[:-1]))
            self.loadJobDirs()
            self.loadDescription()
            self.ui.lab_currentPath.setText(self.currentJob)

    def saveDescription(self):
        text = self.ui.txt_description.toPlainText()
        if serializeJSON(text, self.currentJob):
            self.loadDescription()

    def loadDescription(self):
        data = deserializeJSON(self.currentJob)
        print("Into Load Description")
        if data:
            self.ui.txt_description.setPlainText(data)
        else:
            self.ui.txt_description.setPlainText("""No Data Available. Please save a new Description.\
It writes a <jobname.json> file to job path.""")


def serializeJSON(body, jobpath):
    data_dict = {"message": body}
    file_path = jobpath + "/project_manager.json"
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent = 4)
    except Exception as E:
        print(f"Unable to load Json file")
        hou.ui.displayMessage(f"Unable to load Json file:\n\
             < {E} >", buttons=("ok",))
        return False
    else:
        hou.ui.displayMessage(f"written Desc to File", buttons=("ok",))
        return True


def deserializeJSON(jobpath):
    file_path = jobpath + "/project_manager.json"
    try:
        with open(file_path, 'r') as json_file:
            data_dict = json.load(json_file)
    except FileNotFoundError:
        hou.ui.setStatusMessage(f"{jobpath} >> No description Available", severity = hou.severityType.ImportantMessage)
    except Exception as E:
        hou.ui.displayMessage(f"Unable to load Json file:\n\
             < {E} >", buttons=("ok",))
        return False
    else:
        data = data_dict["message"]
        hou.ui.setStatusMessage("")
        return data
    



def show():
    dialog = ProjectManager()
    dialog.setParent(hou.qt.floatingPanelWindow(None), QtCore.Qt.Window)
    dialog.show()
