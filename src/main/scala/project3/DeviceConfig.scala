package project3

object DeviceConfig:
  var useGPU: Boolean = System.getProperty("scalation.useGPU", "true") != "false"
