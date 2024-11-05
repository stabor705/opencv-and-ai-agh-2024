import logging
import os
from typing import Annotated, Optional

import slicer.util
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import numpy as np
try :
    import cv2
except ImportError:
    os.system("pip install opencv-python")
    import cv2

try :
    import skimage
except ImportError:
    os.system("pip install scikit-image")
from skimage import measure
from skimage import morphology
try :
    import scipy
except ImportError:
    os.system("pip install scipy")
    
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
try:
    from surface_distance.metrics import compute_dice_coefficient
except ImportError:
    # Install surface-distance package by running:
    os.system("git clone https://github.com/deepmind/surface-distance.git")
    slicer.util.pip_install("surface-distance")
    from surface_distance.metrics import compute_dice_coefficient

    
import vtkSegmentationCore

import pandas as pd

#
# LungSegmentation
#




class LungSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("LungSegmentation")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#LungSegmentation">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # LungSegmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="LungSegmentation",
        sampleName="LungSegmentation1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "LungSegmentation1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="LungSegmentation1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="LungSegmentation1",
    )

    # LungSegmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="LungSegmentation",
        sampleName="LungSegmentation2",
        thumbnailFileName=os.path.join(iconsPath, "LungSegmentation2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="LungSegmentation2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="LungSegmentation2",
    )


#
# LungSegmentationParameterNode
#


@parameterNodeWrapper
class LungSegmentationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    outputVolume - The thresholding result.
    referenceVolume - The reference volume for thresholding.
    totalSegVolume - The total segmentation volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode
    referenceVolume: vtkMRMLScalarVolumeNode
    totalSegVolume: vtkMRMLScalarVolumeNode


#
# LungSegmentationWidget
#


class LungSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/LungSegmentation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = LungSegmentationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[LungSegmentationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(), self.ui.referenceSelector.currentNode(), self.ui.totalSegSelector.currentNode())


#
# LungSegmentationLogic
#


class LungSegmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return LungSegmentationParameterNode(super().getParameterNode())

    def dice_coeff(lung, lung_ref):
        # calculate the dice coefficient
        dice = compute_dice_coefficient(np.array(lung).astype(np.bool_), np.array(lung_ref).astype(np.bool_))
        return dice

    def thorax_mask(ct: np.array):
        nib_img = cv2.medianBlur(ct.astype(np.float32), 3)
        _, mask = cv2.threshold(nib_img, -191, 1, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = measure.label(mask, connectivity=1)
        props = measure.regionprops(mask)
        for sl in range(mask.shape[0]):
            max_area = 0
            max_area_idx = 0
            for idx, prop in enumerate(props):
                if prop.area > max_area:
                    max_area = prop.area
                    max_area_idx = idx
            mask[mask != max_area_idx + 1] = 0
            mask[mask == max_area_idx + 1] = 1
        mask = 1 - mask
        return mask.astype(np.uint8)
    
    def lungSegmentation(ct: np.array, mask: np.array):
        _, binarized = cv2.threshold(ct, -320, 1, cv2.THRESH_BINARY_INV)
        binarized = binarized.astype(np.uint8)
        masked = cv2.bitwise_and(binarized, mask)
        blurred = cv2.medianBlur(masked, 5)

        kernel = morphology.ball(3)
        closed = morphology.closing(blurred, footprint=kernel)
        opened = morphology.opening(closed, footprint=kernel)
        filled = binary_fill_holes(opened)

        return filled.astype(np.uint8)

    def performWatershed(segmentedLungs: np.array):
        distance = ndi.distance_transform_edt(segmentedLungs)
        coords = peak_local_max(distance, footprint=np.ones((150, 150, 150)), labels=segmentedLungs)
        mask = np.zeros(distance.shape, dtype=np.bool_)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=segmentedLungs)
        props = measure.regionprops_table(labels, properties=('label', 'area', 'bbox'))
        props_df = pd.DataFrame(props)
        lungs_props_df = props_df.sort_values(by="area", ascending=False).iloc[:2].sort_values(by="bbox-1")
        left_lung_label = lungs_props_df.iloc[0]["label"]
        right_lung_label = lungs_props_df.iloc[1]["label"]
        left_lung = labels == left_lung_label
        right_lung = labels == right_lung_label
        result = np.zeros(labels.shape)
        result[left_lung] = 1
        result[right_lung] = 2
        return result

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                referenceVolume: vtkMRMLScalarVolumeNode,
                totalSegVolume: vtkMRMLScalarVolumeNode,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param referenceVolume: reference volume for thresholding
        :param totalSegVolume: total segmentation volume
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        labelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')

        inputArray = slicer.util.arrayFromVolume(inputVolume)
        mask = LungSegmentationLogic.thorax_mask(inputArray)
        segmentedLungs = LungSegmentationLogic.lungSegmentation(inputArray, mask)
        splitLungs = LungSegmentationLogic.performWatershed(segmentedLungs)

        slicer.util.updateVolumeFromArray(labelMapVolumeNode, splitLungs)
        labelMapVolumeNode.CopyOrientation(inputVolume)
    
        segmentationNode = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(segmentationNode)
        segmentationNode.CreateDefaultDisplayNodes()

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelMapVolumeNode, segmentationNode
        )
        segmentationNode.SetName(inputVolume.GetName() + "_segmentation")

        slicer.mrmlScene.RemoveNode(labelMapVolumeNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

        # save split lungs to output volume
        slicer.util.updateVolumeFromArray(outputVolume, splitLungs)
        outputVolume.SetName(inputVolume.GetName() + "_lungs_split")

        # Dice calculation
        if not referenceVolume or not totalSegVolume:
            raise ValueError("Input or output volume is invalid")
        diceStartTIme = time.time()
        referenceArray = slicer.util.arrayFromVolume(referenceVolume)
        totalSegArray = slicer.util.arrayFromVolume(totalSegVolume)
        segmentedLungsArray = splitLungs

        leftLungReference = referenceArray == 2
        rightLungReference = referenceArray == 3

        splitTotalLungs = LungSegmentationLogic.performWatershed(totalSegArray) # making sure that the total segmentation is split into left and right lungs as well
        # because TotalSegmenter splits lungs into 5 segments
        leftLungTotalSeg = splitTotalLungs == 1
        rightLungTotalSeg = splitTotalLungs == 2


        leftLungSegmented = splitLungs == 1
        rightLungSegmented = splitLungs == 2

        leftLungDice = LungSegmentationLogic.dice_coeff(leftLungSegmented, leftLungReference)
        rightLungDice = LungSegmentationLogic.dice_coeff(rightLungSegmented, rightLungReference)
        totalLungDice = LungSegmentationLogic.dice_coeff(segmentedLungsArray, referenceArray)

        logging.info(f"Left lung dice: {leftLungDice}")
        logging.info(f"Right lung dice: {rightLungDice}")
        logging.info(f"Total lung dice: {totalLungDice}")

        leftLungDiceTotalSeg = LungSegmentationLogic.dice_coeff(leftLungSegmented, leftLungTotalSeg)
        rightLungDiceTotalSeg = LungSegmentationLogic.dice_coeff(rightLungSegmented, rightLungTotalSeg)
        totalLungDiceTotalSeg = LungSegmentationLogic.dice_coeff(segmentedLungsArray, totalSegArray)

        logging.info(f"Left lung dice with total segmentation: {leftLungDiceTotalSeg}")
        logging.info(f"Right lung dice with total segmentation: {rightLungDiceTotalSeg}")
        logging.info(f"Total lung dice with total segmentation: {totalLungDiceTotalSeg}")

        diceStopTime = time.time()

        logging.info(f"Dice calculation completed in {diceStopTime-diceStartTIme:.2f} seconds")



        return segmentationNode


#
# LungSegmentationTest
#


class LungSegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_LungSegmentation1()

    def test_LungSegmentation1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("LungSegmentation1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = LungSegmentationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
