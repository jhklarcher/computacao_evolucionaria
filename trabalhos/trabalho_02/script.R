library(gramEvol)
library(tidyverse)

data <- read_csv("trabalhos/trabalho_02/treino.csv") %>%
  mutate(class = recode(class, "STABLE" = 0, "UP" = 1, "DOWN" = -1))

X <- data %>%
  select(-class, -datetime)

y <- data %>%
  select(class)

ruleDef <- list(expr = grule(op(expr, expr), func(expr), var),
                func = grule(sin, cos),
                op = grule("+", "-", "*", "/"),
                var = grule(as.matrix(X[, 1]),
                            as.matrix(X[, 2]),
                            as.matrix(X[, 3]),
                            as.matrix(X[, 4]),
                            as.matrix(X[, 5]))
                )

grammarDef <- CreateGrammar(ruleDef)

SymRegFitFunc <- function(expr) {

  result <- eval(expr)
  if (any(is.nan(result)) || !is.numeric(mean(abs(y - result))) || any(mean(abs(y - result)))) {
    return(999999999999)
  }
  else {
    return(mean(abs(y - result)))
  }
}



ge <- GrammaticalEvolution(grammarDef, SymRegFitFunc, iterations = 50)



