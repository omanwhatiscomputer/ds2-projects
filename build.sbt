// build.sbt

lazy val scalation = project.in(file("."))

  .settings(

    scalaVersion  := "3.7.2",

    scalacOptions ++= Seq(

       "-deprecation",         // emit warning and location for usages of deprecated APIs

       "-explain",             // explain errors in more detail

       "-new-syntax",          // require `then` and `do` in control expressions.

       "-Wunused:all",         // warn of unused imports, ...

       "-Xfatal-warnings")     // fail the compilation if there are any warnings

//  javacOptions  += "--add-modules jdk.incubator.vector"

  )

fork := true
// Fork the JVM when running

// Enable Panama/FFM API native access
run / javaOptions += "--enable-native-access=ALL-UNNAMED"

// Optional: enable for tests too
Test / fork := true
Test / javaOptions += "--enable-native-access=ALL-UNNAMED"
