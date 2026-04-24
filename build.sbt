// build.sbt

lazy val scalation = project.in(file("."))
  .settings(
    scalaVersion := "3.7.2",

    scalacOptions ++= Seq(
      "-deprecation",
      "-explain",
      "-new-syntax",
      "-Wunused:all",
      "-Xfatal-warnings",
      "-release", "21" // Important: produce Java 21 compatible bytecode
    ),

    // UPDATED: Modern Maven Central coordinates for TornadoVM v2.2.0
    libraryDependencies ++= Seq(
      "io.github.beehive-lab" % "tornado-api" % "2.2.0",
      "io.github.beehive-lab" % "tornado-runtime" % "2.2.0" // <-- THIS WAS MISSING
    ),

    fork := true,
    run / javaOptions ++= Seq(
      "--add-modules=jdk.incubator.vector",
      "--enable-native-access=ALL-UNNAMED",
      // These exports allow TornadoVM to work with JDK 21+ internals
      "--add-exports=jdk.internal.vm.ci/jdk.vm.ci.meta=ALL-UNNAMED",
      "--add-exports=jdk.internal.vm.ci/jdk.vm.ci.runtime=ALL-UNNAMED",
      "-Dtornado.device.desc=true"
    )
  )