import React from 'react'
import { PageHeader } from '@components/ui/PageHeader'
import { Card } from '@components/ui/Card'

export const Ask: React.FC = () => {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Ask"
        description="Ask questions in natural language and get intelligent answers"
      />
      
      <Card>
        <div className="card-content">
          <div className="text-center py-12">
            <h3 className="text-lg font-medium text-neutral-900 mb-2">
              Ask Page
            </h3>
            <p className="text-neutral-600">
              This page will be implemented next with the natural language query interface.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
