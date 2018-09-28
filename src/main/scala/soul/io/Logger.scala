package soul.io

import java.io.{File, PrintWriter}

import scala.collection.mutable.ArrayBuffer

/** Logger to collect info about a undersampling/oversampling process
  *
  * @author Néstor Rodríguez Vico
  */
private[soul] class Logger {
  private[soul] val log: ArrayBuffer[String] = new ArrayBuffer[String](0)

  /** Add a new message to the log
    *
    * @param msg message to store
    */
  private[soul] def addMsg(msg: String): Unit = {
    this.log += msg
  }

  /** Store the logs into a file
    *
    * @param file filename where to store the logs
    */
  private[soul] def storeFile(file: String): Unit = {
    val data = new PrintWriter(new File(file + ".log"))
    this.log.foreach((line: String) => data.write(line + "\n"))
    data.close()
  }
}
