package scalation
package mathstat

import scala.util.Try

/** Single control point for TornadoVM device selection.
 *
 *  CPU vs GPU:
 *    DeviceConfig.useGPU = false          // skip TornadoVM, run pure JVM
 *    DeviceConfig.useGPU = true           // dispatch to TornadoVM
 *
 *  Device selection (driver:device index, only when useGPU = true):
 *    DeviceConfig.driverIndex = 0         // 0 = OpenCL, 1 = PTX (NVIDIA)
 *    DeviceConfig.deviceIndex = 0         // 0 = first device (usually GPU)
 *    DeviceConfig.deviceIndex = 1         // 1 = second device (maybe CPU)
 *
 *  Multithreaded CPU via TornadoVM:
 *    Select the OpenCL CPU device (visible in startup output because
 *    -Dtornado.device.desc=true is already set in build.sbt javaOptions).
 *    TornadoVM's CPU OpenCL backend vectorises and uses all cores automatically.
 *
 *  JVM flag shorthand (set in build.sbt or sbt run command):
 *    -Dscalation.useGPU=false
 *    -Dscalation.driver=0  -Dscalation.device=1
 */
object DeviceConfig:

  var useGPU: Boolean =
    System.getProperty("scalation.useGPU", "true") != "false"

  var driverIndex: Int =
    Try(System.getProperty("scalation.driver", "0").toInt).getOrElse(0)

  var deviceIndex: Int =
    Try(System.getProperty("scalation.device", "0").toInt).getOrElse(0)

  /** TornadoVM per-task device override via system property.
   *  Call this BEFORE building a TaskGraph to route it to the chosen device.
   *  @param graphName  the string passed to `new TaskGraph(graphName)`
   *  @param taskId     the string passed to `.task(taskId, ...)`
   */
  def applyDeviceProperty(graphName: String, taskId: String): Unit =
    System.setProperty(s"$graphName.$taskId.device", s"$driverIndex:$deviceIndex")
