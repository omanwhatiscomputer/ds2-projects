import scala.io.Source

// 1. Read the argfile directly during the SBT build
lazy val tornadoSdkPath = "/home/khubayeeb_k/.sdkman/candidates/tornadovm/4.0.0-jdk25-full/tornado-argfile"
lazy val tornadoVMOptions = Source.fromFile(tornadoSdkPath)
  .getLines()
  .map(_.trim)
  .filter(_.nonEmpty)
  .filterNot(_.startsWith("#"))
  .flatMap(_.split("\\s+"))
  .toSeq
lazy val scalation = project.in(file("."))
  .settings(
    scalaVersion := "3.7.2",

    scalacOptions ++= Seq(
      "-deprecation",
      "-explain",
      "-new-syntax",
      "-Wunused:all",
      "-Xfatal-warnings",
      "-release", "25"
    ),

    libraryDependencies ++= Seq(
      "io.github.beehive-lab" % "tornado-api" % "4.0.0",
      "io.github.beehive-lab" % "tornado-runtime" % "4.0.0"
    ),

    fork := true,

    // 2. Inject the extracted flags + your device target!
    run / javaOptions ++= tornadoVMOptions ++ Seq(
      "-Dtornado.backends=ptx",
      "-Ds0.t0.device=0:0"
    )
  )