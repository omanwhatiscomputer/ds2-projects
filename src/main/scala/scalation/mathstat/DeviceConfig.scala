package scalation.mathstat

object DeviceConfig:
  var useGPU: Boolean = System.getProperty("scalation.useGPU", "true") != "false"
