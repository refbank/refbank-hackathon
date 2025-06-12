DATA_LOC = here("harmonized_data")

get_tbl <- function(con = DATA_LOC, dataset_name, tbl_name) {
  # con is fake rn, and takes in the root dir for all datasets
  if (missing(dataset_name)) {
    stop("dataset_name must be provided")
  }
  if (missing(tbl_name)) {
    stop("tbl_name must be provided")
  }
  
  file_path <- file.path(con, dataset_name, paste0(tbl_name, ".csv"))
  if (!file.exists(file_path)) {
    stop(paste("File does not exist:", file_path))
  }
  
  out <- read_csv(file_path, show_col_types = FALSE)
  
  if (tbl_name == "conditions") {
    out <- out |> 
      mutate(structure = structure |> 
               as.character() |> 
               replace_na("other") |> 
               factor(levels = c("thin", "medium", "med_thick", "thick", "network-swap", "naive-swap")) |> 
               fct_drop())
  }
  out
}

get_trials_full <- function(con = DATA_LOC, dataset_name) {
  trials <- get_tbl(con, dataset_name, "trials") |> 
    mutate(target = as.character(target),
           matchers = as.character(matchers))
  conditions <- get_tbl(con, dataset_name, "conditions")
  
  out <- trials |> 
    left_join(conditions, by = join_by(condition_id))
  out
}

get_messages_full <- function(con = DATA_LOC, dataset_name) {
  messages <- get_tbl(con, dataset_name, "messages") |> 
    rename(player_num = player_id)
  trials <- get_trials_full(con, dataset_name)
  
  out <- trials |> 
    left_join(messages, by = join_by(trial_id))
  out
}

get_choices_full <- function(con = DATA_LOC, dataset_name) {
  choices <- get_tbl(con, dataset_name, "choices") |> 
    rename(player_num = player_id)
  if (nrow(choices) == 0) {
    return(tibble())
  }
  
  trials <- get_trials_full(con, dataset_name)
  
  out <- trials |> 
    left_join(choices, by = join_by(trial_id, option_set))
  out
}
