name := "soul"

version := "1.0.0"

organization := "com.github.soul"

organizationName := "Néstor Rodríguez Vico and David López Pretel"

organizationHomepage := Some(url("https://github.com/NestorRV/soul"))

scalaVersion := "2.12.6"
scalacOptions in(Compile, doc) ++= Opts.doc.title("soul")
scalacOptions += "-deprecation"
scalacOptions += "-unchecked"
scalacOptions += "-feature"

libraryDependencies ++= Seq(
  "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.2",
  "com.paypal.digraph" % "digraph-parser" % "1.0",
  "com.thesamet" %% "kdtree" % "1.0.5",
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"