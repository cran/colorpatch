% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/colorpatch_impl.R
\name{OrderDataTSP}
\alias{OrderDataTSP}
\title{Orders rows and column distances with traveling salesman ordering \link{TSP}}
\usage{
OrderDataTSP(row.dist, col.dist, ...)
}
\arguments{
\item{row.dist}{row distances}

\item{col.dist}{column distances}

\item{...}{optional parameters fed to the \code{\link[TSP:solve_TSP]{TSP::solve_TSP()}} function}
}
\value{
a list with irow and icol containing the orders of rows and columns
}
\description{
Orders rows and column distances with traveling salesman ordering \link{TSP}
}
