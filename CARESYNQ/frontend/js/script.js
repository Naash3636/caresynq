document.addEventListener("DOMContentLoaded", () => {
  const date = document.getElementById("todayDate");
  if (date) date.textContent = new Intl.DateTimeFormat("en-IN", { weekday:"long", day:"numeric", month:"short", year:"numeric" }).format(new Date());
  const dateInput = document.querySelector('input[name="entry_date"]');
  if (dateInput && !dateInput.value) dateInput.value = new Date().toISOString().slice(0,10);
});