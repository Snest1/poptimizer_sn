package services

import (
	"github.com/WLM1ke/poptimizer/data/internal/domain"
	"github.com/WLM1ke/poptimizer/data/pkg/lgr"
)

// ErrorsRule - правило обработки ошибок.
type ErrorsRule struct {
	Logger *lgr.Logger
}

// NewErrorsRule создает правило обработки событий-ошибок.
func NewErrorsRule(logger *lgr.Logger) *ErrorsRule {
	return &ErrorsRule{Logger: logger}
}

// Activate - активирует правило.
//
// Реагирует паникой на событие ошибок и не использует исходящий канал.
func (r *ErrorsRule) Activate(in <-chan domain.Event, _ chan<- domain.Event) {
	r.Logger.Infof("ErrorRule: started")
	defer r.Logger.Infof("ErrorRule: stopped")

	for event := range in {
		event, ok := event.(domain.UpdatedErrHappened)
		if ok {
			r.Logger.Panicf("ErrorsRule: %#v", event)
		}
	}
}
