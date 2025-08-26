import React from 'react'
import { PageHeader } from '@components/ui/PageHeader'
import { Card } from '@components/ui/Card'

export const Models: React.FC = () => {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Models"
        description="Manage AI models, adapters, and training configurations"
      />
      
      <Card>
        <div className="card-content">
          <div className="text-center py-12">
            <h3 className="text-lg font-medium text-neutral-900 mb-2">
              Models Page
            </h3>
            <p className="text-neutral-600">
              This page will be implemented next with model management features.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
