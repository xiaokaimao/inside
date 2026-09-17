# Regenerate fixtures using the unmodified author source (requires R + gtools).
# Run from the repository root: Rscript experiments/verify_shapdoe_r.R
source("third_party/shapdoe/R/estsh.R")
out <- "third_party/shapdoe/fixtures"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
original_ls <- onels
original_coa_prime <- onecoa.prime
original_coa <- onecoa

for (case in c("ls5", "coa5", "coa8", "coa9", "coa7_project6")) {
  draws <- list()
  sample <- function(...) {
    result <- base::sample(...)
    draws[[length(draws)+1]] <<- result
    return(result)
  }
  set.seed(1234)
  if (case == "ls5") {
    q <- 5; real_n <- 5; design <- original_ls(q)
  } else if (case == "coa5" || case == "coa7_project6") {
    q <- ifelse(case == "coa5", 5, 7)
    real_n <- ifelse(case == "coa5", 5, 6)
    design <- original_coa_prime(q)
  } else {
    q <- ifelse(case == "coa8", 8, 9); real_n <- q
    p <- ifelse(q == 8, 2, 3)
    polynomial <- if (q == 8) c(1,0,1,1) else c(1,1,2)
    design <- original_coa(q, p, polynomial)
  }
  write.table(design - 1, file.path(out, paste0(case, "_design.csv")),
              sep=",", row.names=FALSE, col.names=FALSE)
  writeLines(vapply(draws, function(x) paste(x, collapse=","), ""),
             file.path(out, paste0(case, "_draws.txt")))
  utility <- function(sets) {
    mask <- sum(2^(sets[sets <= real_n]-1))
    sin(.37*mask) + cos(.11*mask) + mask^2/10000 - 1
  }
  # Fix each generator to the design just produced so that the original
  # estimator and the Python port evaluate exactly the same permutations.
  onels <- function(d) design
  onecoa.prime <- function(d) design
  onecoa <- function(d,p,f_d) design
  if (case == "ls5") {
    values <- est.shls(q, nrow(design), utility)
  } else if (case == "coa5" || case == "coa7_project6") {
    values <- est.shcoa.prime(q, nrow(design), utility)
  } else {
    values <- est.shcoa(q, nrow(design), utility, p, polynomial)
  }
  write.table(values, file.path(out, paste0(case, "_values.csv")),
              sep=",", row.names=FALSE, col.names=FALSE)
}
writeLines(c(R.version.string, paste("gtools", packageVersion("gtools")),
             paste("RNGkind", paste(RNGkind(), collapse=", "))),
           file.path(out, "runtime.txt"))
