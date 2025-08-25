library(tidyverse)
library(ggthemes)
library(ggplot2)
library(emmeans)

# Load in dataaset
setwd("/Users/ryderfried/Downloads/")
dat <- read.csv("overall_def_total_no_puncsym_v2.csv")


# Re-format/pre-process data
dat_piv <- dat |> pivot_longer(X1:X6, values_to = "words", names_to="round") |>
  group_by(round, speech_type) |> mutate(game_id = row_number())

# Join together all the different types of "other" and "pronoun" and then convert to proportions
edited_dat_piv <- dat_piv |> 
  mutate(speech_type = ifelse((speech_type == "these_those") |
                                (speech_type == "this_that") |
                                (speech_type == "num"), 
                                "other", speech_type)) |> 
  mutate(speech_type = ifelse((speech_type == "pers_pro") |
                                (speech_type == "poss_pro") |
                                (speech_type == "other_pro"), 
                              "pronoun", speech_type)) |> 
  mutate(speech_type = ifelse((speech_type == "indef"),
                                "a",  speech_type)) |>
  group_by(round, game_id, speech_type) |> summarise("words" = sum(words), .groups="drop") |> 
  group_by(round, game_id) |> mutate("prop" = words / sum(words,na.rm=TRUE))



edited_dat_piv$round <- as.numeric(substr(edited_dat_piv$round, 2, nchar(dat_piv$round)))

# Sort speech_type for graphing order
edited_dat_piv$speech_type <- factor(edited_dat_piv$speech_type, 
                              levels = c("bare_noun", "a", "the", "pronoun", "adj", "other")
)

# Omit rounds 4 & 5
edited_dat_piv <- edited_dat_piv[(edited_dat_piv$round != 5 & edited_dat_piv$round != 6),]

# Make spaghetti plot
plt <- ggplot(edited_dat_piv, aes(x=round, y=prop, color=speech_type)) + 
  geom_line(aes(group=interaction(game_id, speech_type)), alpha=0.4) + 
  geom_smooth(aes(group=(speech_type)), method="lm", formula = y ~ log(x), linewidth=2, alpha=0.2, se=TRUE) +
  scale_color_brewer(palette = "Set1",
    name = "Speech Type",
    labels = c(
      "adj" = "Adjective",
      "verbs" = "Verbs",
      "a" = "'A'",
      "the" = "'The'",
      "pronoun" = "Pronoun",
      "other" = "Other",
      "bare_noun" = "Noun"
      )
    ) +
  labs(x="Round", y="Proportion of First Word of Noun Phrases (%)") + 
  theme(legend.position="bottom") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    legend.position = "right",
    legend.title    = element_text(size = 24),
    legend.text     = element_text(size = 22) 
)


# Create dataframe with just 'a' and 'the'
a_the_dat <- edited_dat_piv[(edited_dat_piv$speech_type == "the" | edited_dat_piv$speech_type == "a"),]

# Plot just 'a' and 'the'
a_the_plt <- ggplot(a_the_dat, aes(x=round, y=prop, color=speech_type)) + 
  geom_line(aes(group=interaction(game_id, speech_type)), alpha=0.4) + 
  geom_smooth(aes(group=(speech_type)), method="lm", formula = y ~ log(x), linewidth=2, alpha=0.2, se=TRUE) +
  scale_color_manual(name = "Speech Type",
                     values = c(
                       "a"     = "#377EB8",
                       "the"       = "#4DAF4A"),
                     labels = c(
                       "a" = "A",
                       "the" = "'The'"
                     ) # Same colors as 'Set1' from 'plt'
  ) +
  labs(x="Round", y="Proportion of First Word of Noun Phrases (%)") + 
  theme(legend.position="bottom") +
  theme_minimal() +
  theme(
  axis.title.x = element_text(size = 12),   # X-axis title
  axis.title.y = element_text(size = 12),
  legend.position = "right",
  legend.title    = element_text(size = 24),  # title bigger
  legend.text     = element_text(size = 22)   # item labels bigger
  )

# Build model

edited_dat_piv_analyze <- edited_dat_piv |> mutate("log_round" = log(round))

model <- lm(prop ~ (log_round * speech_type), data = edited_dat_piv_analyze)
summary(model)

# p_tests
trends <- emtrends(model, ~ speech_type, var="log_round")
contrast(trends, method = "pairwise") # look at a-the interaction
summary(trends, infer = TRUE) # look at bare_noun


# Test log v linear model for bare noun with R^2
bare_noun_test_dat <- edited_dat_piv_analyze[(edited_dat_piv_analyze$speech_type == "bare_noun"),]
log_model <- lm(prop ~ log_round, data = bare_noun_test_dat)
summary(log_model)
lm_model <- lm(prop ~ round, data = bare_noun_test_dat)
summary(lm_model)


             
             



