library(tidyverse)
library(ggthemes)
library(ggplot2)
library(emmeans)

# Load in dataset
setwd("/Users/ryderfried/Downloads/")
dat <- read.csv("overall_syn_con_total_v2.csv")

# Re-organize/pre-process data
dat_piv <- dat |> pivot_longer(X1:X6, values_to = "words", names_to="round") 
dat_piv$round <- as.numeric(substr(dat_piv$round, 2, nchar(dat_piv$round)))


new_dat_piv <- dat_piv |> group_by(round, speech_type) |> mutate("game_id" = row_number())
new_dat_piv <- new_dat_piv[!is.na(new_dat_piv$words),]

# Put auxiliary as a part of verbs
edited_dat_piv <- new_dat_piv |> mutate(speech_type = ifelse(speech_type == "auxiliary", "verbs", speech_type)) |>
  group_by(round, game_id, speech_type) |> summarise("words" = sum(words), .groups="drop") 

# Remove punctuation, symbols, spaces, and end of line and then convert to proportions
edited_dat_piv <- edited_dat_piv[!((edited_dat_piv$speech_type == "punctuation") | 
                                     (edited_dat_piv$speech_type == "symbols") |
                                     (edited_dat_piv$speech_type == "end_of_line") |
                                     (edited_dat_piv$speech_type == "spaces")
                                   ),] |> 
  group_by(round, game_id) |> mutate("prop" = words / sum(words,na.rm=TRUE)) |>
  group_by(round, speech_type) |> summarise("total_prop" = mean(prop), .groups="drop") 

edited_dat_piv <- edited_dat_piv[(edited_dat_piv$round != 5 & edited_dat_piv$round != 6),]


num_vars = length(unique(edited_dat_piv$speech_type)) # number of unique speech_types
num_keep = 6 # number of speech_types to display in graphs

# Order df so its speech_types are in descending order by their round 1 proportions
speech_type_round_one_df <- filter(edited_dat_piv, round == 1)|> arrange(total_prop)
speech_type_round_one_lst <- rev(speech_type_round_one_df$speech_type)



# Move least-popular by round 1 proportion into 'other' category
rel_speech_types <- speech_type_round_one_lst[1:num_keep]
irrel_speech_types <- speech_type_round_one_lst[(num_keep + 1):num_vars]
df_rel <- edited_dat_piv[edited_dat_piv$speech_type %in% rel_speech_types,]
df_irrel <- edited_dat_piv[edited_dat_piv$speech_type %in% irrel_speech_types,]
irrel_grouped_rows <- df_irrel |> group_by(round) |> summarise("total_prop" = sum(total_prop), "speech_type" = "other") 
full_df_ordered <- bind_rows(irrel_grouped_rows, df_rel) 

# Order df by speech_type first round proportion from least to most (but so 'other' is first) for nice graphing
speech_type_round_one_df_v2 <- filter(full_df_ordered, round == 1)|> arrange(total_prop)
other_row <- speech_type_round_one_df_v2[speech_type_round_one_df_v2$speech_type == "other",]
rows_without_other <- speech_type_round_one_df_v2[speech_type_round_one_df_v2$speech_type != "other",]
ordered_type_df <- rbind(other_row, rows_without_other)
speech_type_round_one_lst_v2 <- ordered_type_df$speech_type
full_df_ordered$speech_type <- factor(full_df_ordered$speech_type, levels = speech_type_round_one_lst_v2)

# Build log model for p value tests

full_df_ordered_analyze <- full_df_ordered |> mutate("log_round" = log(round))

model <- lm(total_prop ~ log_round*speech_type, data=full_df_ordered_analyze)
trends <- emtrends(model, ~ speech_type, var="log_round")
summary(trends, infer = TRUE) # look at nouns and determiners


# Area plot
area_plot <- ggplot(full_df_ordered, aes(x = round, y = total_prop, fill = speech_type, group=speech_type)) +
  geom_area(alpha=0.6 , linewidth=0.25, colour="black") +
  scale_fill_brewer(palette = "Set1", direction=-1) +
  ggthemes::theme_few() +
  scale_x_continuous(breaks = c(1,2,3,4))+
  theme(aspect.ratio = 2.5) +
  ylab('% words')

# Bar plot
bar_plot <- ggplot(full_df_ordered, aes(x = round, y = total_prop, fill = speech_type, group=speech_type)) +
  geom_col(position="stack", alpha=0.6 , linewidth=0.25, colour="black") +
  scale_fill_brewer(palette = "Set1", direction=-1) +
  ggthemes::theme_few() +
  scale_x_continuous(breaks = c(1,2,3,4))+
  theme(aspect.ratio = 2.5) +
  ylab('% words') +
  theme(
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    legend.position = "right",
    legend.title    = element_text(size = 24),
    legend.text     = element_text(size = 22) 
  )