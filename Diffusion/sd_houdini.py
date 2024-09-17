import hou
import os
from hutil.Qt import QtCore, QtWidgets, QtUiTools

#local path to the script
scriptpath =  os.path.dirname(__file__)

class HouDiffusion(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(1200, 600)
        #load UI file
        loader = QtUiTools.QUiLoader()
        self.ui = loader.load(f"{scriptpath}/ui_diffusion.ui")
        
        #SET LABEL
        self.ui.node_lbl.setText(f"Node:{self.sel_node().name()}")
        self.ui.refresh_btn.clicked.connect(self.refresh)
                
        # get geo attribs
        self.geo = self.sel_node().geometry()
        self.attribs = []
        self.point_values = []
        self.get_point_attribs()
        
        # layout
        mainLayout = QtWidgets.QVBoxLayout()    #generate a layout for houdini
        mainLayout.setContentsMargins(0,0,0,0)  
        mainLayout.addWidget(self.ui)   #add ui to widget
        self.setLayout(mainLayout)
        
    
    def refresh(self):
        self.geo = self.sel_node().geometry()
        print(self.geo)
        self.attribs = []
        self.point_values = []
        self.ui.data_tbl.clear()
        self.ui.data_tbl.setRowCount(0)
        self.ui.data_tbl.setColumnCount(0)
        self.get_point_attribs()

    
    def get_point_attribs(self):
        point_attribs = self.geo.pointAttribs()
        att_suffix = ['x', 'y', 'z']

        for pt_attrib in point_attribs:
            point_attr = None
            if self.geo.findPointAttrib(pt_attrib.name()) != None:
                point_attr = self.geo.findPointAttrib(pt_attrib.name())
                if point_attr.size() > 1:
                    for i in range(point_attr.size()):
                        self.ui.data_tbl.insertColumn(i)
                        self.attribs.append(f"{point_attr.name()}.{att_suffix[i]}") # P.x P.y P.z
                else:
                    self.ui.data_tbl.insertColumn(1)
                    self.attribs.append(point_attr.name())
        
        #create rows
        for i in range(len(self.geo.points())):
                self.ui.data_tbl.insertRow(i)
        
        col_labels = tuple(self.attribs)
        self.ui.data_tbl.setHorizontalHeaderLabels(col_labels)

        row_labels = range(len(self.geo.points()))
        row_labels = tuple(map(str, row_labels))
        
        #get points data
        for point in self.geo.points():
                #iterate on attribs
                for row_num, pt_attrib in enumerate(point_attribs):
                    point_attr = None
                    if self.geo.findPointAttrib(pt_attrib.name()) != None:
                        point_attr = self.geo.findPointAttrib(pt_attrib.name())
                        if point_attr.size() > 1:
                            for i in range(point_attr.size()):
                                self.point_values.append(point.attribValue(point_attr)[i])
                        else:
                            self.point_values.append(point.attribValue(point_attr))
        
        #set row labels
        self.ui.data_tbl.setVerticalHeaderLabels(row_labels)
        
        #populate data table
        for col, attrib in enumerate(self.attribs):
            for row in range(len(self.geo.points())):
                index = (row * len(self.attribs)) + col
                data = float("%.6f" %self.point_values[index])
                item = QtWidgets.QTableWidgetItem(str(data))
                self.ui.data_tbl.setItem(row, col, item)

    def sel_node(self):
        try: 
            return hou.selectedNodes()[0]
        except Exception:
            hou.ui.displayMessage("sel a node")


def show():
      dialog = HouDiffusion()
      dialog.setParent(hou.qt.floatingPanelWindow(None), QtCore.Qt.Window)
      dialog.show()