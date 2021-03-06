% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/colorpatch_impl.R
\name{ComputeSymmetry}
\alias{ComputeSymmetry}
\title{Computes the symmetry of a given bi-variate color palette}
\usage{
ComputeSymmetry(pal, color.space = "LAB")
}
\arguments{
\item{pal}{A two-sided input palette \link[colorspace:color-class]{colorspace::color}}

\item{color.space}{Color space where the distances shall be computed (default "LAB")}
}
\value{
a data frame with index, side (pos/neg) and distance
}
\description{
Computes the symmetry of a given bi-variate color palette
}
\examples{
data("OptimGreenRedLAB")
df <- ComputeSymmetry(OptimGreenRedLAB)
print(df)
}
