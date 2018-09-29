name := "soul"

version := "1.0.0"

organization := "com.github.soul"

organizationName := "Nestor Rodriguez Vico and David Lopez Pretel"

organizationHomepage := Some(url("https://github.com/NestorRV/soul"))

scalaVersion := "2.12.6"
scalacOptions in(Compile, doc) ++= Opts.doc.title("soul")
scalacOptions += "-deprecation"
scalacOptions += "-unchecked"
scalacOptions += "-feature"

libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.2"
libraryDependencies += "com.paypal.digraph" % "digraph-parser" % "1.0"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"