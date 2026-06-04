// Mint-C - Web-based digit recognition tool with C++ CNN backend
// Copyright (C) 2026 Chrollis
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

export default {
  app: {
    title: "Mint-C",
    drawer: {
      dashboard: "Dashboard",
      datasets: "Datasets",
      evaluate: "Evaluate",
      misc: "Misc",
      model: "Model",
      recognize: "Recognize",
      train: "Train",
    },
    footer: {
      ready: "Ready, waiting for action",
      trainingPrefix: "Training",
      training: "Training: Epoch {epoch}/{total}, Loss {loss}",
      finished: "Training finished",
      cancelled: "Training cancelled",
      error: "Training error",
      idle: "Idle",
    },
  },

  common: {
    cancel: "Cancel",
    confirm: "Confirm",
    close: "Close",
    save: "Save",
    delete: "Delete",
    clear: "Clear",
    refresh: "Refresh",
    upload: "Upload",
    download: "Download",
    start: "Start",
    loading: "Loading...",
    modelLoaded: "Model loaded",
    modelNotLoaded: "Model not loaded",
    noDataset: "Please select a dataset",
    noData: "No data",
    modelNotReady: "Model not loaded, please load a model first",
    average: "Average",
    clearHistory: "Clear History",
    clearHistoryText:
      "Are you sure you want to clear all history? This action cannot be undone.",
    confirmDeletion: "Confirm Deletion",
    realtimeLoss: "Real-time Loss",
    realtimeAccuracy: "Real-time Accuracy (%)",
    historyLoss: "History Loss",
    historyAccuracy: "History Accuracy (%)",
    loss: "Loss",
    accuracy: "Accuracy (%)",
    digit: "Digit",
    offsetLabel: "Offset",
    countLabel: "Count",
  },

  dashboard: {
    serverStatus: "Server Status",
    serviceOk: "Service OK",
    serviceError: "Service Error",
    uptime: "Uptime: {time}",
    modelStatus: "Model Status",
    modelSyncHint: "Model status synced from backend",
    trainingStatus: "Training Status",
    trainingIdle: "No training task",
    trainingIdleHint: 'Go to "Train" page to start a new training',
    projectDescription: "Project Description",
    readmeLoadError:
      "Failed to load README. Please check network or file existence.",
    trainingHistory: "Training History",
  },

  datasets: {
    uploadTitle: "Upload New Dataset",
    datasetNameLabel: "Dataset Name",
    datasetNamePlaceholder: "e.g., my_mnist",
    imagesFileLabel: "images.idx3-ubyte file",
    labelsFileLabel: "labels.idx1-ubyte file",
    listTitle: "Dataset List",
    statusOk: "OK",
    statusError: "Error",
    deleteConfirmText:
      "Are you sure you want to delete the following {count} dataset(s)?",
    deleteIrreversible: "This action cannot be undone.",
    downloadConfirmTitle: "Confirm Download",
    downloadConfirmText:
      "Are you sure you want to download the following {count} dataset(s)?",
    singleDeleteConfirmText:
      'Are you sure you want to delete dataset "{name}"? This action cannot be undone.',
    mergeBtn: "Merge",
    mergeDialogTitle: "Merge Datasets",
    mergeDialogText:
      "Merge the following {count} dataset(s) into a new dataset:",
    mergeTargetLabel: "New Dataset Name",
    mergeTargetPlaceholder: "Enter name for the merged dataset",
    uploadSuccess: 'Dataset "{name}" uploaded successfully',
    downloadStart: 'Downloading dataset "{name}"',
    batchDownloadSuccess: "Packed {count} dataset(s) for download",
    deleteSuccess: "Deleted {count} dataset(s)",
    singleDeleteSuccess: 'Dataset "{name}" deleted',
    mergeSuccess: "Merge successful, new dataset: {name}",
    errors: {
      loadFailed: "Failed to load dataset list",
      missingName: "Please enter a dataset name",
      missingFiles: "Please select both images and labels files",
      uploadFailed: "Upload failed, please check network or file format",
      downloadFailed: 'Failed to download dataset "{name}"',
      batchDownloadFailed:
        "Batch download failed, please check network and retry",
      partialDeleteFailed: "Some datasets failed to delete",
      singleDeleteFailed: 'Failed to delete dataset "{name}"',
      mergeFailed: "Merge failed",
    },
    headers: {
      name: "Name",
      sampleCount: "Sample Count",
      status: "Status",
      actions: "Actions",
    },
  },

  evaluate: {
    title: "Model Evaluation",
    selectDataset: "Select Dataset",
    batchSizeLabel: "Batch Size",
    statisticsTitle: "Evaluation Statistics",
    perClassHistoryTitle: "Per-class Historical Accuracy (%)",
    perClassName: "Run {index}",
    startSuccess: "Evaluation started",
    cancelSuccess: "Evaluation cancelled",
    historyCleared: "History cleared",
    errors: {
      startFailed: "Failed to start evaluation",
      cancelFailed: "Failed to cancel evaluation",
    },
  },

  layerParams: {
    out_channels: "Output channels",
    kernel_radius: "Kernel radius",
    stride: "Stride",
    padding: "Padding",
    init: "Weight init",
    kernel_size: "Kernel size",
    pool_type: "Pool type",
    out_features: "Output features",
    keep_prob: "Keep prob",
    eps: "Epsilon",
    momentum: "Momentum",
    activation: "Activation",
  },

  layerTypes: {
    convolve: "Convolution",
    pool: "Pooling",
    flatten: "Flatten",
    dense: "Dense",
    dropout: "Dropout",
    batchnorm: "BatchNorm",
    activate: "Activation",
    softmax: "Softmax",
  },

  misc: {
    serverConfigTitle: "Server Configuration",
    portLabel: "Port",
    logDirLabel: "Log Directory",
    tempDirLabel: "Temp Directory",
    datasetDirLabel: "Dataset Directory",
    systemLogsTitle: "System Logs",
    logLevelLabel: "Level",
    logCountLabel: "Count",
    logLevels: {
      debug: "Debug",
      info: "Info",
      warning: "Warning",
      error: "Error",
    },
    aboutTitle: "About Mint-C",
    aboutDesc:
      "Web-based digit recognition tool with C++ CNN backend for inference and training.",
    copyright: "Copyright © 2026 Chrollis",
    licenseMsg:
      "This program is free software released under the GNU General Public License v3.0.",
    licenseLinkText: "View GPLv3 License",
    thirdPartyNotice:
      "This project uses the following open source components: Eigen, cpp-httplib, nlohmann/json, stb, Vue 3, Vuetify, etc., each under their respective licenses.",
    showDetails: "Show Details",
    hideDetails: "Hide Details",
    configRefreshSuccess: "Configuration refreshed",
    configLoadFailed: "Failed to load configuration",
    configSaveSuccess: "Configuration saved",
    configSaveFailed: "Failed to save configuration",
    logsLoadFailed: "Failed to load logs",
  },

  model: {
    currentArchTitle: "Current Model Architecture",
    copyToClipboardBtn: "Copy to Clipboard",
    loadModelBtn: "Load Model",
    newArchTitle: "New Model Architecture",
    loadFromClipboardBtn: "Load from Clipboard",
    applyBtn: "Apply to Current",
    addLayerBtn: "Add Layer",
    inputShapeLabel: "Input Shape",
    outputShapeLabel: "Output Shape",
    addLayerDialogTitle: "Add Layer",
    layerTypeLabel: "Layer Type",
    selectLayerType: "Select",
    selectOption: "Select",
    noParams: "No parameters",
    addBtn: "Add",
    shapeValidationError: "Output shape must be {expected}, got {actual}",
    shapeValidationFailed: "Output shape validation failed",
    applySuccess: "Model architecture applied and saved",
    applyFailed: "Failed to apply model",
    loadCurrentFailed: "Failed to load current model",
    copySuccess: "Architecture copied to clipboard",
    copyFailed: "Copy failed",
    loadClipboardSuccess: "Loaded from clipboard successfully",
    loadClipboardInvalid: "Invalid clipboard content",
    loadClipboardFailed: "Failed to load from clipboard",
    loadModelSuccess: "Model loaded successfully",
    loadModelFailed: "Failed to load model",
    downloadStarted: "Model download started",
    downloadFailed: "Model download failed",
    confirmClearNewModelTitle: "Clear New Model",
    confirmClearNewModelText:
      "Are you sure you want to clear all layers in the new model? This action cannot be undone.",
    clearSuccess: "New model cleared",
  },

  modelLayerList: {
    noLayers: "No layers",
    noParams: "No parameters",
    totalLayers: "Total {count} layers",
  },

  recognize: {
    canvasTitle: "Canvas",
    resultTitle: "Recognition Results",
    recognizeBtn: "Recognize",
    confidenceLabel: "Confidence",
    positionLabel: "Position: x={x}, y={y}, w={w}, h={h}",
    relabelTitle: "Relabel",
    noResultHint:
      'No recognition results. Draw or upload an image and click "Recognize".',
    addToPendingBtn: "Add to Pending",
    batchRelabelBtn: "Batch Relabel",
    selectAllBtn: "Select All",
    pendingTitle: "Pending Samples",
    saveToDatasetBtn: "Save to Dataset",
    noPendingHint: "No pending samples",
    relabelDialogTitle: "Relabel Digit",
    batchRelabelDialogTitle: "Batch Relabel",
    correctDigitLabel: "Correct Digit",
    currentDigitHint: "Current recognition result: {digit}",
    batchRelabelHint: "Will relabel {count} selected digit(s)",
    saveDatasetDialogTitle: "Save to Dataset",
    datasetNameLabel: "Dataset Name",
    datasetNamePlaceholder: "e.g., my_handwritten",
    saveDatasetHint: "Will save {count} sample(s) to this dataset.",
    sampleRelabelDialogTitle: "Modify Sample Label",
    currentSampleLabelHint: "Current label: {label}",
    confirmDeletePendingText:
      "Are you sure you want to remove this sample? This action cannot be undone.",
    confirmClearPendingTitle: "Clear All Pending Samples",
    confirmClearPendingText:
      "Are you sure you want to clear all pending samples? This action cannot be undone.",
  },

  train: {
    configTitle: "Training Configuration",
    selectDataset: "Select Dataset",
    loadingHyperparams: "Loading hyperparameters...",
    monitorTitle: "Training Monitor",
    progressWhileEpoch: " while Epoch",
    finishDialogTitle: "Training Finished",
    finishDialogText: "Training completed. Accept this model?",
    discardBtn: "Discard",
    acceptBtn: "Accept",
    startSuccess: "Training started",
    cancelSuccess: "Training cancelled",
    errors: {
      startFailed: "Failed to start training",
      cancelFailed: "Failed to cancel training",
    },
    hyperparams: {
      learning_rate: "Learning Rate",
      batch_size: "Batch Size",
      epochs: "Epochs",
    },
  },
};
